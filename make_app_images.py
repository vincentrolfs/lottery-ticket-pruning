import json
import random

import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# %%

SEED = 123898123
AMOUNT = 100

# %% Load data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_loader_normalized = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='data/CIFAR10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    ),
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='data/CIFAR10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    ),
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# %%

random.seed(SEED)
np.random.seed(SEED)
indices = np.random.permutation(len(val_loader.dataset))
images = []
labels = []

counter = 0
stats = {}

for index in indices:
    img, label = val_loader_normalized.dataset[index]
    if label not in stats:
        stats[label] = 0

    if stats[label] >= AMOUNT // 10:
        continue

    images.append(img.tolist())
    labels.append(label)
    stats[label] += 1

    save_image(val_loader.dataset[index][0], "html/app/images/{}.png".format(counter))

    counter += 1

    if counter >= AMOUNT:
        break

print(stats)

f = open('html/app/js/images.js', 'w+')
f.write('var images = ' + json.dumps(images) + '.map(i => arrayToTypedArray(i));')
f.write('\n')
f.write('var trueLabels = ' + json.dumps(labels) + ';')
f.close()
