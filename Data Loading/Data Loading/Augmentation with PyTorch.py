import torchvision.transforms as transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader

# 1. custom_transform
custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])

# 2. Load dataset
dataset = datasets.ImageFolder('./images_dataSAT', transform=custom_transform)

# 3. Class names/index
print("Classes:", dataset.classes)

# 4. Batch retrieval/shape
loader = DataLoader(dataset, batch_size=8, shuffle=True)
dataiter = iter(loader)
images, labels = next(dataiter)
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# 5. Display images
import matplotlib.pyplot as plt
grid_img = torchvision.utils.make_grid(images)
plt.imshow(grid_img.permute(1,2,0))
plt.show()
