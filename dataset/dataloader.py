import os
from create_dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    ),
])

def data_transforms(image_path , mask_path):

    file_list = os.listdir(image_path)
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(image_folder=image_path, mask_folder=mask_path, file_list=train_files, transform=transform)
    val_dataset = CustomDataset(image_folder=image_path, mask_folder=mask_path, file_list=val_files, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader , val_loader


