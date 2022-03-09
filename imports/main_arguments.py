import argparse

__all__ = ['parser']

from imports.utils import PruningType

parser = argparse.ArgumentParser(description='ResNet56 for CIFAR10 in pytorch')

parser.add_argument('--validate-only', dest='validate_only', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('--progress', default='', type=str, metavar='PATH',
                    help='path to progress (default: none)')
parser.add_argument('--type',
                    default=PruningType.LT_TRADITIONAL.value,
                    dest='pruning_type',
                    type=str,
                    help='which type of pruning'
                    )

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu-only', dest='cpu_only', action='store_true',
                    help='Only use CPU (use this when no GPU/Cuda is available)')

parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves progress at every specified number of epochs',
                    type=int, default=10)

parser.add_argument('--compression-ratio', dest='compression_ratio', type=float,
                    help='compression ratio (original_size/new_size)')
parser.add_argument('--iterative-round', dest='iterative_round', default=1, type=int,
                    help='which round of iterative pruning')
parser.add_argument('--seed-index', dest='seed_index', type=int,
                    help='seed')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--learning-rate', dest='lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='data/models', type=str)
parser.add_argument('--cifar-dir', dest='cifar_dir',
                    help='Where to save CIFAR10 dataset',
                    type=str, default='data/CIFAR10')
