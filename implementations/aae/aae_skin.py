import argparse
import os
import numpy as np
import math
import itertools

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
    "--latent_dim", type=int, default=100, help="dimensionality of the latent code"
)
parser.add_argument(
    "--img_size",
    type=int,
    default=int(os.getenv("img_size", 28)),
    help="size of each image dimension",
)
parser.add_argument(
    "--channels", type=int, default=int(os.getenv("channels", 1)), help="number of image channels"
)
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval between image sampling"
)
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

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Configure data loader
dataset_init = datasets.ImageFolder(
    root=opt.data_path, transform=util.custom_preprocessing(opt), is_valid_file=util.is_valid_file
)

# find the classes to augment
to_aug = util.get_number_instances_to_aug(dataset_init)

for class_idx, data in to_aug.items():

    if "all" == class_idx:
        continue

    # this AAE learns one class at a time
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
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adversarial_loss(
                discriminator(encoded_imgs), valid
            ) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

        # for each epoch, save sample images
        util.sample_image(
            to_aug=to_aug,
            latent_dim=opt.latent_dim,
            generator=decoder,
            results_path=opt.results_path,
            all_labels=False,
            one_label=data["class"],
        )
