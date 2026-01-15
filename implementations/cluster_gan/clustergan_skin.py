from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image

    from itertools import chain as ichain

except ImportError as e:
    print(e)
    raise ImportError

import util

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument(
    "--n_epochs",
    type=int,
    default=int(os.getenv("n_epochs", 200)),
    help="number of epochs of training",
)
parser.add_argument(
    "--batch_size", type=int, default=int(os.getenv("batch_size", 64)), help="size of the batches"
)
parser.add_argument(
    "-i",
    "--img_size",
    dest="img_size",
    type=int,
    default=int(os.getenv("img_size", 128)),
    help="Size of image dimension",
)
parser.add_argument(
    "--channels", type=int, default=int(os.getenv("channels", 1)), help="number of image channels"
)
parser.add_argument(
    "--n_classes",
    type=int,
    default=int(os.getenv("n_classes", 10)),
    help="number of classes for dataset",
)
parser.add_argument(
    "--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation"
)
parser.add_argument(
    "-d",
    "--latent_dim",
    dest="latent_dim",
    default=100,
    type=int,
    help="Dimension of latent space",
)
parser.add_argument(
    "-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate"
)
parser.add_argument(
    "-c",
    "--n_critic",
    dest="n_critic",
    type=int,
    default=5,
    help="Number of training steps for discriminator per iter",
)
parser.add_argument(
    "-w", "--wass_flag", dest="wass_flag", action="store_true", help="Flag for Wasserstein metric"
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
parser.add_argument(
    "--test_data_path",
    type=str,
    default=os.getenv("test_data_path", ""),
    help="path to dataset folder with subfolders as labels",
)

opt = parser.parse_args()

# Training details
n_epochs = opt.n_epochs
batch_size = opt.batch_size

lr = opt.learning_rate
b1 = 0.5
b2 = 0.9
decay = 2.5 * 1e-5
n_skip_iter = opt.n_critic

# Data dimensions
img_size = opt.img_size
channels = opt.channels
# Latent space info
latent_dim = opt.latent_dim
n_c = opt.n_classes
betan = 10
betac = 10

# Wasserstein+GP metric flag
wass_metric = opt.wass_flag

x_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):

    assert fix_class == -1 or (fix_class >= 0 and fix_class < n_c), (
        "Requested class %i outside bounds." % fix_class
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Sample noise as generator input, zn
    zn = Variable(
        Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad
    )

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = LongTensor(shape).fill_(0)

    if fix_class == -1:
        zc_idx = zc_idx.random_(n_c)
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.0)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

    zc = Variable(zc_FT, requires_grad=req_grad)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "shape={}".format(self.shape)


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """

    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = "generator"
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        channels, img_size, _ = x_shape

        # Calcular las dimensiones intermedias basadas en el tamaño de imagen final
        # Dos ConvTranspose2d con stride=2 multiplican por 4 (2*2)
        # img_size = ishape_h * 4, entonces ishape_h = img_size // 4
        ishape_h = img_size // 4
        ishape_w = img_size // 4
        self.ishape = (128, ishape_h, ishape_w)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=False),
            # Reshape to 128 x (ishape_h x ishape_w)
            Reshape(self.ishape),
            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = "encoder"
        self.channels = opt.channels
        self.latent_dim = latent_dim
        self.n_c = n_c

        # Calcular las dimensiones intermedias basadas en el tamaño de imagen
        # Dos Conv2d con stride=2 dividen por 4 (2*2)
        # Para conv2d(kernel=4, stride=2, padding=0): output = (input - 4) // 2 + 1
        # Primera capa: (img_size - 4) // 2 + 1
        # Segunda capa: ((img_size - 4) // 2 + 1 - 4) // 2 + 1
        # Simplificado para img_size=28: (28-4)//2+1 = 13, (13-4)//2+1 = 5
        img_size = opt.img_size
        h1 = (img_size - 4) // 2 + 1
        cshape_h = (h1 - 4) // 2 + 1
        cshape_w = cshape_h
        self.cshape = (128, cshape_h, cshape_w)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            # Flatten
            Reshape(self.lshape),
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, latent_dim + n_c),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0 : self.latent_dim]
        zc_logits = z[:, self.latent_dim :]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """

    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = "discriminator"
        self.channels = opt.channels

        # Calcular las dimensiones intermedias basadas en el tamaño de imagen
        # Igual que el Encoder
        img_size = opt.img_size
        h1 = (img_size - 4) // 2 + 1
        cshape_h = (h1 - 4) // 2 + 1
        cshape_w = cshape_h
        self.cshape = (128, cshape_h, cshape_w)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose

        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            # Flatten
            Reshape(self.lshape),
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, 1),
        )

        # If NOT using Wasserstein metric, final Sigmoid
        if not self.wass:
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity


# Loss function
bce_loss = torch.nn.BCELoss()
xe_loss = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator_CNN(latent_dim, n_c, x_shape)
encoder = Encoder_CNN(latent_dim, n_c)
discriminator = Discriminator_CNN(wass_metric=wass_metric)

if cuda:
    generator.cuda()
    encoder.cuda()
    discriminator.cuda()
    bce_loss.cuda()
    xe_loss.cuda()
    mse_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Configure data loader
dataset_init = datasets.ImageFolder(
    root=opt.data_path,
    transform=util.custom_preprocessing(opt),
    is_valid_file=util.is_valid_file,
)

# find the classes to augment
to_aug = util.get_number_instances_to_aug(dataset_init)

dataset = datasets.ImageFolder(
    root=opt.data_path,
    transform=util.custom_preprocessing(opt),
    is_valid_file=util.is_valid_file,
)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Test data loader
testdataset = datasets.ImageFolder(
    root=opt.test_data_path,
    transform=util.custom_preprocessing(opt),
    is_valid_file=util.is_valid_file,
)

testdata = DataLoader(
    testdataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_imgs, test_labels = next(iter(testdata))
test_imgs = Variable(test_imgs.type(Tensor))

ge_chain = ichain(generator.parameters(), encoder.parameters())

optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# ----------
#  Training
# ----------
ge_l = []
d_l = []

c_zn = []
c_zc = []
c_i = []

# Training loop
print("\nBegin training session with %i epochs...\n" % (n_epochs))
for epoch in range(n_epochs):

    for i, (imgs, itruth_label) in enumerate(dataloader):

        # Ensure generator/encoder are trainable
        generator.train()
        encoder.train()

        # Zero gradients for models
        generator.zero_grad()
        encoder.zero_grad()
        discriminator.zero_grad()

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------------
        #  Train Generator + Encoder
        # ---------------------------

        optimizer_GE.zero_grad()

        # Sample random latent variables
        zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c)

        # Generate a batch of images
        gen_imgs = generator(zn, zc)

        # Step for Generator & Encoder, n_skip_iter times less than for discriminator
        if i % n_skip_iter == 0:
            # Discriminator output from generated samples
            D_gen = discriminator(gen_imgs)

            # Encode the generated images
            enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

            # Calculate losses for z_n, z_c
            zn_loss = mse_loss(enc_gen_zn, zn)
            zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

            # Check requested metric
            if wass_metric:
                # Wasserstein GAN loss
                ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
            else:
                # Vanilla GAN loss
                valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                v_loss = bce_loss(D_gen, valid)
                ge_loss = v_loss + betan * zn_loss + betac * zc_loss

            ge_loss.backward()
            optimizer_GE.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate new images for discriminator training (detached from generator graph)
        zn_d, zc_d, _ = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c)
        gen_imgs_d = generator(zn_d, zc_d).detach()

        # Discriminator output from real and generated samples
        D_real = discriminator(real_imgs)
        D_gen_d = discriminator(gen_imgs_d)

        # Measure discriminator's ability to classify real from generated samples
        if wass_metric:
            # Gradient penalty term
            grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs_d)

            # Wasserstein GAN loss w/gradient penalty
            d_loss = torch.mean(D_real) - torch.mean(D_gen_d) + grad_penalty

        else:
            # Vanilla GAN loss
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            real_loss = bce_loss(D_real, valid)
            fake_loss = bce_loss(D_gen_d, fake)
            d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    # Save training losses
    d_l.append(d_loss.item())
    ge_l.append(ge_loss.item())

    # Generator in eval mode
    generator.eval()
    encoder.eval()

    # Set number of examples for cycle calcs
    n_sqrt_samp = 5
    n_samp = n_sqrt_samp * n_sqrt_samp

    ## Cycle through test real -> enc -> gen
    t_imgs, t_label = test_imgs.data, test_labels
    # Encode sample real instances
    e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
    # Generate sample instances from encoding
    teg_imgs = generator(e_tzn, e_tzc)
    # Calculate cycle reconstruction loss
    img_mse_loss = mse_loss(t_imgs, teg_imgs)
    # Save img reco cycle loss
    c_i.append(img_mse_loss.item())

    ## Cycle through randomly sampled encoding -> generator -> encoder
    zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp, latent_dim=latent_dim, n_c=n_c)
    # Generate sample instances
    gen_imgs_samp = generator(zn_samp, zc_samp)

    # Encode sample instances
    zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

    # Calculate cycle latent losses
    lat_mse_loss = mse_loss(zn_e, zn_samp)
    lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)

    # Save latent space cycle losses
    c_zn.append(lat_mse_loss.item())
    c_zc.append(lat_xe_loss.item())

    # Save cycled and generated examples!
    r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
    e_zn, e_zc, e_zc_logits = encoder(r_imgs)
    reg_imgs = generator(e_zn, e_zc)

    ## Generate samples for specified classes
    for idx in range(n_c):

        # dont need to aug the greater classes
        if idx not in to_aug:
            continue

        # Sample specific class
        zn_samp, zc_samp, zc_samp_idx = sample_z(
            shape=to_aug[idx]["images"], latent_dim=latent_dim, n_c=n_c, fix_class=idx
        )

        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)

        # Save class-specified generated examples!
        util.save_images(
            gen_imgs_samp,
            os.path.join(opt.results_path, to_aug[idx]["class"]),
        )

    print(
        "[Epoch %d/%d] \n"
        "\tModel Losses: [D: %f] [GE: %f]" % (epoch, n_epochs, d_loss.item(), ge_loss.item())
    )

    print(
        "\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]"
        % (img_mse_loss.item(), lat_mse_loss.item(), lat_xe_loss.item())
    )
