import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *

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
    "--batch_size", type=int, default=int(os.getenv("batch_size", 8)), help="size of the batches"
)
parser.add_argument(
    "--dataset_name", type=str, default="img_align_celeba", help="name of the dataset"
)
parser.add_argument(
    "--lr", type=float, default=float(os.getenv("lr", 0.0002)), help="adam: learning rate"
)
parser.add_argument(
    "--b1",
    type=float,
    default=float(os.getenv("b1", 0.5)),
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=float(os.getenv("b2", 0.999)),
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=int(os.getenv("n_cpu", 8)),
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=int(os.getenv("latent_dim", 100)),
    help="dimensionality of the latent space",
)
parser.add_argument(
    "--img_size",
    type=int,
    default=int(os.getenv("img_size", 128)),
    help="size of each image dimension",
)
parser.add_argument(
    "--mask_size", type=int, default=int(os.getenv("mask_size", 32)), help="size of random mask"
)
parser.add_argument(
    "--channels", type=int, default=int(os.getenv("channels", 3)), help="number of image channels"
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=int(os.getenv("sample_interval", 500)),
    help="interval between image sampling",
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

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(input_shape)
discriminator = Discriminator(input_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((opt.img_size // 4, opt.img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Configure data loader
dataset_init = datasets.ImageFolder(
    root=opt.data_path, transform=util.custom_preprocessing(opt), is_valid_file=util.is_valid_file
)

# find the classes to augment
to_aug = util.get_number_instances_to_aug(dataset_init)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def apply_random_mask(imgs):
    idx = np.random.randint(0, opt.img_size - opt.mask_size, (imgs.shape[0], 2))

    masked_imgs = imgs.clone()
    for i, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size
        masked_imgs[i, :, y1:y2, x1:x2] = -1

    return masked_imgs


for class_idx, data in to_aug.items():

    if "all" == class_idx:
        continue

    save_results_path = os.path.join(opt.results_path, data["class"])
    os.makedirs(save_results_path, exist_ok=True)

    dataloader = DataLoader(
        util.ImageDatasetCCGAN(
            opt.data_path,
            transforms_x=transforms_,
            transforms_lr=transforms_lr,
            filter_classes=[data["class"]],
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    for epoch in range(opt.n_epochs):
    
        saved_samples = {}

        for i, batch in enumerate(dataloader):
            imgs = batch["x"]
            imgs_lr = batch["x_lr"]

            masked_imgs = apply_random_mask(imgs)

            # Adversarial ground truths
            valid = Variable(
                Tensor(imgs.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                Tensor(imgs.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False
            )

            if cuda:
                imgs = imgs.type(Tensor)
                imgs_lr = imgs_lr.type(Tensor)
                masked_imgs = masked_imgs.type(Tensor)

            real_imgs = Variable(imgs)
            imgs_lr = Variable(imgs_lr)
            masked_imgs = Variable(masked_imgs)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(masked_imgs, imgs_lr)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            # Save first ten samples
            if not saved_samples:
                saved_samples["imgs"] = real_imgs.clone()
                saved_samples["masked"] = masked_imgs.clone()
                saved_samples["lowres"] = imgs_lr.clone()
            else:
                saved_samples["imgs"] = torch.cat((saved_samples["imgs"], real_imgs), 0)
                saved_samples["masked"] = torch.cat((saved_samples["masked"], masked_imgs), 0)
                saved_samples["lowres"] = torch.cat((saved_samples["lowres"], imgs_lr), 0)

            batches_done = epoch * len(dataloader) + i

        # End epoch - Generate inpainted image
        gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
        # Save sample
        sample = gen_imgs.data
        util.save_images(images=sample, path=save_results_path)
