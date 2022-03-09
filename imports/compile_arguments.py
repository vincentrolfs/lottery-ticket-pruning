import argparse

__all__ = ['parser']

parser = argparse.ArgumentParser(description='Export pruned ResNet-56 models to JavaScript')

parser.add_argument('--model', required=True, default='', type=str, metavar='PATH',
                    help='path to model')
parser.add_argument('--output', dest='output_dir',
                    help='The directory used to output the compiled model',
                    default='data/models_js', type=str)
parser.add_argument('--skip-write', dest='skip_write', action='store_true',
                    help='If set, output file is not written')
parser.add_argument('--skip-log', dest='skip_log', action='store_true',
                    help='If set, logging information is supressed')
