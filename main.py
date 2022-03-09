# %% Imports

import json
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from imports import utils
from imports.main_arguments import parser
from imports.prune import prune
from imports.resnet import resnet56
from imports.train import train
from imports.utils import get_cuda_func, new_progress, load_parameters, \
    save_parameters, PruningType, print_size, ParameterType
from imports.validate import validate

# %% Setup

# This is needed because some of the models where pickled when the import was just "utils"
# Now it is "imports.utils"
sys.modules['utils'] = utils

args = parser.parse_args()
cuda = get_cuda_func(args.cpu_only)
pruning_type = PruningType.from_str(args.pruning_type)

ALL_SEEDS = [11696672, 15713537, 36569120, 70206358, 75504233, 83494940, 90478944, 92519636, 95284986, 96488735]

# %% Load data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=args.cifar_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=args.cifar_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    ),
    batch_size=128,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)

# %% Initialize model + cuda

model = torch.nn.DataParallel(resnet56())
cuda(model)

cudnn.benchmark = True

# %% Load or create progress

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.progress:
    if os.path.isfile(args.progress):
        print("=> loading progress '{}'".format(args.progress))
        progress = load_parameters(model, args.progress, args.cpu_only)
        print("=> loaded progress '{}'. data: {}".format(args.progress, progress))
    else:
        print("=> no progress found at '{}'".format(args.progress))
        print("=> exiting.")
        sys.exit()
else:
    progress = new_progress(args.compression_ratio, ALL_SEEDS[args.seed_index], args.iterative_round)
    print("=> starting with progress data: {}".format(progress))

# %% Check arguments

for key in ["seed", "compression_ratio"]:
    if key not in progress or progress[key] is None:
        print("No {} found!".format(key))
        sys.exit()

if progress['compression_ratio'] < 1:
    print("compression_ratio must be at least 1!")
    sys.exit()

# %% Perform seeding

print("Seeding with {}".format(progress['seed']))
np.random.seed(progress['seed'])
torch.manual_seed(progress['seed'])

# %% Define loss

criterion = cuda(nn.CrossEntropyLoss())

if args.half:
    model.half()
    criterion.half()

# %% Validate

if args.validate_only:
    fname = args.save_dir + '/validations.json'

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            scores = json.load(f)
    else:
        scores = {}

    scores[args.progress] = validate(val_loader, model, criterion, cuda, args)

    with open(fname, 'w+') as f:
        json.dump(scores, f)

    sys.exit()

# %% Define optimizer

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150],
                                                    last_epoch=progress['next_epoch'] - 1)

# %% Perform pruning

all_masks = None

if progress["next_epoch"] == 0 and progress["compression_ratio"] > 1:
    print("Applying LT pruning")
    relative_size, all_masks = prune(args, cuda, model, progress)

    print("New size: {:.2f}%".format(100 * relative_size))
    print_size(model)

# %% Train loop

save_parameters(model, progress, args.save_dir, pruning_type)
progress['type'] = ParameterType.CHECKPOINT

while progress['next_epoch'] < args.epochs:
    # save checkpoint
    if progress['next_epoch'] > 0 and progress['next_epoch'] % args.save_every == 0:
        print("Saved progress: {}".format(progress))
        save_parameters(model, progress, args.save_dir, pruning_type)

    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(train_loader, model, criterion, optimizer, progress['next_epoch'], cuda, args, all_masks)
    lr_scheduler.step()

    # evaluate on validation set
    accuracy = validate(val_loader, model, criterion, cuda, args)

    print(accuracy)

    # remember best prec@1 and prec@5
    progress['all_accuracies'].append(accuracy)

    # go to next epoch
    progress['next_epoch'] += 1

progress['type'] = ParameterType.FINAL
save_parameters(model, progress, args.save_dir, pruning_type)
