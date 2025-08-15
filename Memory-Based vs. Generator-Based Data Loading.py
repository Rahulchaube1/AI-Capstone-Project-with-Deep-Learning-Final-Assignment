# Import libraries
import os
from PIL import Image
import matplotlib.pyplot as plt

# 1. Find image shape
img = Image.open('./images_dataSAT/class_0_non_agri/0.jpg')
print("Shape:", img.size)

# 2. Display first 4 non-agri images
non_agri_dir = './images_dataSAT/class_0_non_agri/'
files = sorted(os.listdir(non_agri_dir))[:4]
for fname in files:
    img = Image.open(os.path.join(non_agri_dir, fname))
    plt.imshow(img)
    plt.title(fname)
    plt.show()

# 3. agri_images_paths list
dir_agri = './images_dataSAT/class_1_agri/'
agri_images_paths = sorted([os.path.join(dir_agri, f) for f in os.listdir(dir_agri)])
print("Agri images:", agri_images_paths[:2])

# 4. Count agricultural images
print("Total agri images:", len(agri_images_paths))

# 5. Display first 4 agri images
for fname in agri_images_paths[:4]:
    img = Image.open(fname)
    plt.imshow(img)
    plt.title(fname)
    plt.show()
