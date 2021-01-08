import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import random

model = torchvision.models.__dict__['vgg19']()
print(model)

img = torch.rand(1,3,256,256)
out = model.features(img)
print(out.size())

import torchvision.transforms as trans

crop = trans.RandomCrop(224)
img = torch.rand(1,3,256,256)

out = crop(img)
print(out.size())

def equally_divide_patches(img, divide_num):
    patche_size_w = int(img.size[0] / divide_num) 
    patche_size_h = int(img.size[1] / divide_num)

    patches = []
    for cnt_i, i in enumerate(range(0, img.size[1], patche_size_h)):
        if cnt_i == divide_num:
            break
        for cnt_j, j in enumerate(range(0, img.size[0], patche_size_w)):
            if cnt_j == divide_num:
                break
            box = (j, i, j+patche_size_w, i+patche_size_h)
            patches.append(img.crop(box))

    return patches

def display_images(
    images: [Image], 
    columns=3, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):

    if not images:
        print("No images to display.")
        return 

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(3, 3, i + 1)
        plt.imshow(image)

    plt.show()

image = Image.open("/mnt/hdd02/ShanghaiTech/part_A/train_data/images/IMG_7.jpg").convert("RGB")

p = equally_divide_patches(image, 3)
print(len(p))

display_images(p)

def create_pos_pair(patches):
    idx = random.randint(0, len(patches)-1)
    img1 = patches[idx]
    img2 = patches[idx]
    label = 1
    return img1, img2, label

def create_neg_pair(patches):
    idx = random.sample(range(0, len(patches)-1), k=2)
    img1 = patches[idx[0]]
    img2 = patches[idx[1]]
    label = 0
    return img1, img2, label

def get_img(img):
    patches = equally_divide_patches(img, 3)

    if random.random() > 0.5:
        img1, img2, label = create_pos_pair(patches)
    else:
        img1, img2, label = create_neg_pair(patches)

    return img1, img2, label

res = []
for i in range(10):
    img1, img2, label = get_img(image)
    flag = False
    if img1 == img2:
        flag = True
    res.append([flag, label])

print(res)