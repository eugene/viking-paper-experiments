from typing import Literal

# import jax
# import jax.numpy as jnp
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets

# import torchvision
# from PIL import Image
# from torchvision import datasets
from torchvision import transforms as T

from vp.helper import set_seed


def collate_fn(batch):
    data, target = zip(*batch)

    data = torch.stack(data).permute(0, 2, 3, 1).numpy()
    target = torch.stack(target).numpy() + 0.0

    return {"image": data, "label": target}


def Imagenette_loaders(
    batch_size: int = 128,
    purp: Literal["train", "sample"] = "train",
    seed: int = 0,
    n_samples_per_class=None,
):
    n_classes = 10

    if purp == "train":
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif purp == "sample":
        train_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    test_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def target_transform(y):
        return F.one_hot(torch.tensor(y), n_classes)

    train_dataset = datasets.Imagenette(
        root="data",
        download=True,
        transform=train_transform,
        target_transform=target_transform,
    )

    val_dataset = datasets.Imagenette(
        root="data",
        download=True,
        transform=test_transform,
        target_transform=target_transform,
    )

    set_seed(seed)
    train_subset, _ = torch.utils.data.random_split(train_dataset, [9469 - 256, 256])
    set_seed(seed)
    _, val_subset = torch.utils.data.random_split(val_dataset, [9469 - 256, 256])

    train_loader = data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
        collate_fn=collate_fn,
    )

    val_loader = data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
        collate_fn=collate_fn,
    )

    ### Test loader (Imagenette validation set)
    test_dataset = datasets.Imagenette(
        root="data",
        download=True,
        split="val",
        transform=test_transform,
        target_transform=target_transform,
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
