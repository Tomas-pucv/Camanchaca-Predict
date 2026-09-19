"""Transformaciones de imagen.

IMPORTANTE: no usar jitter de color / brillo — alterarían la apariencia de la
niebla y por tanto la "señal" que el modelo debe aprender. Solo aumentación
geométrica.
"""
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool = True, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
