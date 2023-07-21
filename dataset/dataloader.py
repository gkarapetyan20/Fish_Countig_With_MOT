import os
from cudtom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Set your desired transformations (e.g., resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and masks to a consistent size
    transforms.ToTensor(),           # Convert images and masks to tensors
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def build_dataloader(image_folder , mask_folder):
    # Get a list of file names in the dataset
    file_list = os.listdir(image_folder)

    # Split the dataset into training and validation subsets
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # Create training and validation datasets
    train_dataset = CustomDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=transform)
    val_dataset = CustomDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=transform)

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader


