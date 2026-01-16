import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import util

from dotenv import load_dotenv

load_dotenv()

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs",
    type=int,
    default=int(os.getenv("n_epochs", 200)),
    help="number of epochs of training",
)
parser.add_argument(
    "--batch_size", type=int, default=int(os.getenv("batch_size", 64)), help="size of the batches"
)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient"
)
parser.add_argument(
    "--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient"
)
parser.add_argument(
    "--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation"
)
parser.add_argument(
    "--latent_dim", type=int, default=62, help="dimensionality of the latent space"
)
parser.add_argument(
    "--img_size",
    type=int,
    default=int(os.getenv("img_size", 32)),
    help="size of each image dimension",
)
parser.add_argument(
    "--channels", type=int, default=int(os.getenv("channels", 1)), help="number of image channels"
)
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
parser.add_argument(
    "--data_path",
    type=str,
    default=os.getenv("data_path", ""),
    help="path to dataset folder with subfolders as labels",
)
parser.add_argument(
    "--results_path",
    type=str,
    default=os.getenv("results_path", ""),
    help="path to dataset folder with subfolders as labels",
)
# not used, but needed for sharing arguments
parser.add_argument(
    "--n_classes",
    type=int,
    default=int(os.getenv("n_classes", 10)),
    help="number of classes for dataset",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default=os.getenv("test_data_path", ""),
    help="path to dataset folder with subfolders as labels",
)
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings**2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt


# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, opt.batch_size / 64.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding


# Configure data loader
dataset_init = datasets.ImageFolder(
    root=opt.data_path, transform=util.custom_preprocessing(opt), is_valid_file=util.is_valid_file
)

# find the classes to augment
to_aug = util.get_number_instances_to_aug(dataset_init)

for class_idx, data in to_aug.items():

    if "all" == class_idx:
        continue

    # this EBGAN learns one class at a time
    # we are not generating the greatest classes only (nevus, for example)
    dataset = util.ImageFolderFilterClasses(
        root=opt.data_path,
        transform=util.custom_preprocessing(opt),
        is_valid_file=util.is_valid_file,
        filter_classes=[data["class"]],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Training
    # Reconstruction loss of AE
    pixelwise_loss = nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        pixelwise_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            recon_imgs, img_embeddings = discriminator(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(
                img_embeddings
            )

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_recon, _ = discriminator(real_imgs)
            fake_recon, _ = discriminator(gen_imgs.detach())

            d_loss_real = pixelwise_loss(real_recon, real_imgs)
            d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

            d_loss = d_loss_real
            if (margin - d_loss_fake.data).item() > 0:
                d_loss += margin - d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

        # for each epoch, save sample images
        util.sample_image(
            to_aug=to_aug,
            latent_dim=opt.latent_dim,
            generator=generator,
            results_path=opt.results_path,
            all_labels=False,
            one_label=data["class"],
        )
