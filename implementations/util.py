import os
import torchvision.transforms as transforms

def is_valid_file(filepath: str):
    # Add your custom logic here to determine if the file is valid
    # For example, you might want to check file extensions or other criteria
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.png')
    filename = os.path.basename(filepath)

    return (not filename.startswith(".")) and filename.lower().endswith(valid_extensions)

def custom_preprocessing(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels)
    ])
    return transform