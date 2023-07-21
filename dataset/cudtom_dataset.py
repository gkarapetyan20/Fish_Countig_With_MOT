import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, file_list, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # Load image and mask
        image_path = os.path.join(self.image_folder, filename)
        mask_path = os.path.join(self.mask_folder, filename)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transformations (if any)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask