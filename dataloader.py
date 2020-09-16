from glob import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class XxxDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        super(XxxDataset, self).__init__()
        self.images = []
        self.transforms = transforms
        for path in glob(data_dir + "/*"):
            images = glob(path + "/*")
            self.images.extend(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.images[idx].split("/")[1]
        if "0" in label:
            category = torch.tensor(0).long()
            xx = torch.tensor(0).long()
        if "1" in label:
            category = torch.tensor(1).long()
            xx = torch.tensor(1).long()
        if "2" in label:
            category = torch.tensor(2).long()
            xx = torch.tensor(2).long()
        if "3" in label:
            category = torch.tensor(3).long()
            xx = torch.tensor(2).long()
        if "4" in label:
            category = torch.tensor(4).long()
            xx = torch.tensor(2).long()
        if "5" in label:
            category = torch.tensor(5).long()
            xx = torch.tensor(2).long()
        if "man" in label:
            gender = torch.FloatTensor([1.0, 0.0, 0.0])
        elif "woman" in label:
            gender = torch.FloatTensor([0.0, 1.0, 0.0])
        elif "cartoon" in label:
            gender = torch.FloatTensor([0.0, 0.0, 1.0])
        else:
            gender = torch.FloatTensor([0.0, 0.0, 0.0])

        if self.transforms is not None:
            image = self.transforms(image)
        return image, category, gender, xx


def prepare_data_generator(params):
    train_transform = transforms.Compose(
        [
            transforms.Resize(299, interpolation=2),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(
                brightness=params["data_augmentation"]["brightness"],
                contrast=params["data_augmentation"]["contrast"],
                saturation=params["data_augmentation"]["contrast"],
                hue=params["data_augmentation"]["hue"],
            ),
            transforms.ToTensor(),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(299, interpolation=2),
            transforms.RandomCrop(299),
            transforms.ToTensor(),
        ]
    )

    train_data = XxxDataset(params["training"]["data_folder"], transforms=train_transform)
    val_data = XxxDataset(params["validation"]["data_folder"], transforms=val_test_transform)
    test_data = XxxDataset(params["test"]["data_folder"], transforms=val_test_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=params["training"]["batch_size"],
        shuffle=True,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=True,
        drop_last=False,
        timeout=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=params["validation"]["batch_size"],
        shuffle=True,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=True,
        drop_last=False,
        timeout=0,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=params["test"]["batch_size"],
        shuffle=True,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=True,
        drop_last=False,
        timeout=0,
    )

    return train_loader, val_loader, test_loader
