import torch
from torchvision import transforms
from torch.utils.data import Dataset


class VeinDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
