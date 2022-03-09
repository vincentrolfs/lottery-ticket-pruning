import math
import sys
from os import path

import torch
import torch.nn as nn
from jsmin import jsmin

from imports import utils
from imports.compile_arguments import parser
from imports.resnet import resnet56
from imports.utils import load_parameters

# This is needed because some of the models where pickled when the import was just "utils"
# Now it is "imports.utils"
sys.modules['utils'] = utils

output_log = False


def linear_to_js(linear: nn.Linear, input_zero_channels: set, name):
    zero_channels = set()
    code = [
        'function {}(v) {{'.format(name),
        'return new Float32Array([',
    ]
    multiply_adds = 0

    if output_log: print("processing linear: {}. output channels: {}.".format(name, linear.out_features))

    for out_channel in range(linear.out_features):
        bias = linear.bias[out_channel]
        is_zero_channel = bias == 0
        code.append('({})'.format(bias))

        for in_channel in range(linear.in_features):
            weight = linear.weight[out_channel][in_channel]

            if weight != 0 and in_channel not in input_zero_channels:
                is_zero_channel = False
                code[-1] += '+ ({}) * v[{}]'.format(weight, in_channel)
                multiply_adds += 1

        if is_zero_channel:
            zero_channels.add(out_channel)

        code[-1] += ","

    code.append(']);')
    code.append('}')

    return code, zero_channels, multiply_adds


def _build_conv2d_conditional(shiftX, shiftY, input_shape, value_sum):
    conditions = []
    for dim, var, shift in ((1, 'x', shiftX), (2, 'y', shiftY)):
        if shift > 0:
            conditions.append('{var} + {shift} < {limit}'.format(var=var, shift=shift, limit=input_shape[dim]))
        if shift < 0:
            conditions.append('{var} {shift} >= 0'.format(var=var, shift=shift))

    if len(conditions) == 0:
        return ' + ' + value_sum

    return ' + (({cond}) ? ({value_sum}) : 0)'.format(cond=" && ".join(conditions), value_sum=value_sum)


def _build_conv2d_value(weight, in_channel, shiftX, shiftY, input_shape):
    return '({}) * v[{} + (x + ({}))*{} + (y + ({}))]'.format(
        weight,
        in_channel * input_shape[1] * input_shape[2],
        shiftX,
        input_shape[2],
        shiftY
    )


def conv2d_to_js(conv2d: nn.Conv2d, input_shape, input_zero_channels: set, name):
    assert conv2d.in_channels == input_shape[0]

    output_shape = [
        conv2d.out_channels,
        math.ceil(input_shape[1] / conv2d.stride[0]),
        math.ceil(input_shape[2] / conv2d.stride[1]),
    ]
    zero_channels = set()
    code = [
        'function {}(v) {{'.format(name),
        'const a = new Float32Array({});'.format(output_shape[0] * output_shape[1] * output_shape[2]),
        'for (let x = 0; x < {}; x += {}) {{'.format(input_shape[1], conv2d.stride[0]),
        'for (let y = 0; y < {}; y += {}) {{'.format(input_shape[2], conv2d.stride[1]),
        'let p = x/{};'.format(conv2d.stride[0]),
        'let q = y/{};'.format(conv2d.stride[1]),
    ]
    multiply_adds = 0

    if output_log: print("processing conv2d: {}. output: {}, stride: {}.".format(name, output_shape, conv2d.stride))

    for out_channel in range(output_shape[0]):
        code.append('a[{} + {}*p + q] = ('.format(out_channel * output_shape[1] * output_shape[2], output_shape[2]))
        is_zero_channel = True

        for shiftX in range(conv2d.kernel_size[0]):
            for shiftY in range(conv2d.kernel_size[1]):
                channel_values = []

                for in_channel in range(conv2d.in_channels):
                    weight = conv2d.state_dict()["weight"][out_channel][in_channel][shiftX][shiftY]

                    if weight != 0 and in_channel not in input_zero_channels:
                        channel_values.append(
                            _build_conv2d_value(
                                weight,
                                in_channel,
                                shiftX - conv2d.kernel_size[0] // 2,
                                shiftY - conv2d.kernel_size[1] // 2,
                                input_shape
                            )
                        )
                        multiply_adds += 1

                if len(channel_values) != 0:
                    is_zero_channel = False
                    code.append(
                        _build_conv2d_conditional(
                            shiftX - conv2d.kernel_size[0] // 2,
                            shiftY - conv2d.kernel_size[1] // 2,
                            input_shape,
                            " + ".join(channel_values)
                        )
                    )

        if is_zero_channel:
            code.append('0')
            zero_channels.add(out_channel)

        code.append(');')

    code.append('}')
    code.append('}')
    code.append('return a;')
    code.append('}')

    multiply_adds *= (input_shape[1] // conv2d.stride[0]) * (input_shape[2] // conv2d.stride[1])

    return code, zero_channels, multiply_adds


def _build_batchnorm_term(channel, is_zero_input, mean, weight, var, eps, bias, input_shape):
    mult = weight / torch.sqrt(var + eps)
    add = -mean * mult + bias

    if mult == 0 or is_zero_input:
        return '{}'.format(add)

    if add == 0:
        return '({}) * v[{} + {}*x + y]'.format(mult, channel * input_shape[1] * input_shape[2], input_shape[2])

    return '({}) * v[{} + {}*x + y] + ({})'.format(mult, channel * input_shape[1] * input_shape[2], input_shape[2], add)


def batchnorm2d_to_js(b2d: nn.BatchNorm2d, input_shape, input_zero_channels, name):
    assert b2d.num_features == input_shape[0]
    zero_channels = set()

    code = [
        'function {}(v) {{'.format(name),
        'const a = new Float32Array({});'.format(b2d.num_features * input_shape[1] * input_shape[2]),
        'for (let x = 0; x < {}; x++){{'.format(input_shape[1]),
        'for (let y = 0; y < {}; y++){{'.format(input_shape[2])
    ]
    multiply_adds = 0

    if output_log: print("processing batchnorm: {}. output channels: {}.".format(name, b2d.num_features))

    for channel in range(b2d.num_features):
        code.append('a[{} + {}*x + y] = ('.format(channel * input_shape[1] * input_shape[2], input_shape[2]))

        term = _build_batchnorm_term(
            channel,
            channel in input_zero_channels,
            b2d.running_mean[channel],
            b2d.weight[channel],
            b2d.running_var[channel],
            b2d.eps,
            b2d.bias[channel],
            input_shape
        )

        if term == '0.0':
            term = '0'
            zero_channels.add(channel)
        else:
            multiply_adds += 1

        code.append(term)

        code.append(');')

    code.append('}')
    code.append('}')
    code.append('return a;')
    code.append('}')

    multiply_adds *= input_shape[1] * input_shape[2]

    return code, zero_channels, multiply_adds


def compute_shortcut_zero_channels(young_zero_channels: set, young_shape, old_zero_channels: set, old_shape):
    if young_shape[0] == old_shape[0]:
        assert young_shape[1] == old_shape[1]
        assert young_shape[2] == old_shape[2]
        return young_zero_channels.intersection(old_zero_channels)

    assert young_shape[0] == 2 * old_shape[0]
    assert 2 * young_shape[1] == old_shape[1]
    assert 2 * young_shape[2] == old_shape[2]

    zero_channels = set()

    for channel in young_zero_channels:
        if channel < young_shape[0] // 4 or \
                (channel >= 3 * (young_shape[0] // 4)) or \
                (channel - young_shape[0] // 4) in old_zero_channels:
            zero_channels.add(channel)

    return zero_channels


class CodeBuilder:
    def __init__(self):
        self.code = []
        self.multiply_adds = 0

    def add(self, data):
        code, zero_channels, multiply_adds = data

        self.code += code
        self.multiply_adds += multiply_adds

        return zero_channels

    def note_computation(self, multiply_adds):
        self.multiply_adds += multiply_adds

    def get_full(self):
        return "\n".join(self.code)

    def get_mini(self):
        return jsmin(self.get_full())

    def write_full(self):
        return self._write(self.get_full())

    def write_mini(self):
        return self._write(self.get_mini())

    def _write(self, content):
        fname = self._compute_filename()
        if output_log: print("Writing to {}".format(fname))
        f = open(fname, 'w+')
        f.write(content)
        f.close()

    def _compute_filename(self):
        basename = path.basename(args.model)
        name = basename[:basename.rfind('.')]

        return path.join(args.output_dir, name + ".js")


if __name__ == '__main__':
    args = parser.parse_args()
    output_log = not args.skip_log

    true_model = resnet56()
    model = nn.DataParallel(true_model)
    load_parameters(model, args.model, True)
    true_model.eval()
    model.eval()

    code = CodeBuilder()

    zero_channels = code.add(conv2d_to_js(true_model.conv1, [3, 32, 32], set(), "conv1"))
    zero_channels = code.add(batchnorm2d_to_js(true_model.bn1, [16, 32, 32], zero_channels, 'bn1'))

    old_amount_channels, old_img_size = amount_channels, img_size = 16, 32

    for layer_num in range(1, 4):
        layer = getattr(true_model, "layer" + str(layer_num))

        for block_index in range(len(layer)):
            old_zero_channels = zero_channels

            zero_channels = code.add(conv2d_to_js(
                layer[block_index].conv1,
                [amount_channels, img_size, img_size],
                zero_channels,
                'layer{}_{}_conv1'.format(layer_num, block_index)
            ))

            if block_index == 0 and layer_num >= 2:
                old_amount_channels, old_img_size = amount_channels, img_size
                amount_channels, img_size = 2 * amount_channels, img_size // 2

            zero_channels = code.add(batchnorm2d_to_js(
                layer[block_index].bn1,
                [amount_channels, img_size, img_size],
                zero_channels,
                'layer{}_{}_bn1'.format(layer_num, block_index)
            ))

            zero_channels = code.add(conv2d_to_js(
                layer[block_index].conv2,
                [amount_channels, img_size, img_size],
                zero_channels,
                'layer{}_{}_conv2'.format(layer_num, block_index)
            ))

            zero_channels = code.add(batchnorm2d_to_js(
                layer[block_index].bn2,
                [amount_channels, img_size, img_size],
                zero_channels,
                'layer{}_{}_bn2'.format(layer_num, block_index)
            ))

            zero_channels = compute_shortcut_zero_channels(
                zero_channels,
                [amount_channels, img_size, img_size],
                old_zero_channels,
                [old_amount_channels, old_img_size, old_img_size]
            )
            code.note_computation(amount_channels * img_size * img_size)

            old_amount_channels, old_img_size = amount_channels, img_size

    code.add(linear_to_js(true_model.linear, zero_channels, 'linear'))

    if not args.skip_write:
        code.write_mini()

    print(code.multiply_adds)
