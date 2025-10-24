from vp.data import all_datasets as datasets

train_loader, val_loader, test_loader = datasets.get_dataloaders(
    "Imagenette",
    train_batch_size=32,
    val_batch_size=32,
    purp="train",
    seed=0,
)

batch = next(iter(train_loader))  # type: ignore
