from vp.models.celeba_linen import VAE as CelebaVAE
from vp.models.convnet import ConvNet
from vp.models.densenet import DenseNet
from vp.models.fmnist_vae_linen import VAE as FMNISTVAE
from vp.models.inception import GoogleNet
from vp.models.lenet import LeNet
from vp.models.mlp import MLP
from vp.models.resnet_18 import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from vp.models.resnet_small import (
    ResNet_small,
    ResNetBlock_small,
)
from vp.models.simple import make_conv_classifier, make_mlp, make_mlp_classifier
from vp.models.unet import Unet
from vp.models.utils import load_model
from vp.models.vit import VisionTransformer

MODELS_DICT = {
    "MLP": MLP,
    "LeNet": LeNet,
    "ResNet_small": ResNet_small,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "DenseNet": DenseNet,
    "GoogleNet": GoogleNet,
    "VisionTransformer": VisionTransformer,
    "celeba_vae": CelebaVAE,
    "fmnist_vae": FMNISTVAE,
}

MODELS_WITH_DROPOUT = frozenset(
    {
        "VisionTransformer",
    }
)
