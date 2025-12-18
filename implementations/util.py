import torchvision.transforms as transforms

def custom_preprocessing(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels)
    ])
    return transform