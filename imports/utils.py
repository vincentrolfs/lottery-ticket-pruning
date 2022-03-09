import os
from enum import Enum, unique

import torch


def get_cuda_func(_cpu_only):
    if _cpu_only:
        return lambda x: x

    return lambda x: x.cuda()


@unique
class ParameterType(Enum):
    INITIAL = "initial"
    CHECKPOINT = "checkpoint"
    FINAL = "final"


@unique
class PruningType(Enum):
    LT_TRADITIONAL = "lt_traditional"  # Lottery ticket pruning with large final approach, resetting to initial

    @staticmethod
    def from_str(label):
        if label == PruningType.LT_TRADITIONAL.value:
            return PruningType.LT_TRADITIONAL
        else:
            raise NotImplementedError


def new_progress(_compression_ratio, _seed, _iterative_round):
    return {
        'type': ParameterType.INITIAL,
        'compression_ratio': _compression_ratio,
        'iterative_round': _iterative_round,
        'next_epoch': 0,
        'all_accuracies': [],
        'seed': _seed
    }


def save_parameters(_model, _progress: dict, _save_dir: str, _pruning_type: PruningType):
    progress_copy = _progress.copy()
    print("Saving progress and parameters")
    print("{}".format(progress_copy))
    progress_copy['state_dict'] = _model.state_dict()

    torch.save(progress_copy, get_filename(_save_dir,
                                           progress_copy['seed'],
                                           progress_copy['compression_ratio'],
                                           progress_copy['type'],
                                           _pruning_type,
                                           progress_copy['iterative_round']
                                           ))


def load_parameters(_model, _filename: str, _cpu_only: bool):
    progress = get_progress_and_state_dict(_filename, _cpu_only)
    _model.load_state_dict(progress['state_dict'])

    del progress['state_dict']

    return progress


def get_progress_and_state_dict(_filename: str, _cpu_only: bool):
    return torch.load(_filename, map_location=torch.device('cpu')) if _cpu_only else torch.load(_filename)


def get_state_dict(_save_dir: str, _seed: int, _compression_ratio: float, _type: ParameterType, _cpu_only: bool,
                   _pruning_type: PruningType, _iterative_round: int):
    filename = get_filename(_save_dir, _seed, _compression_ratio, _type, _pruning_type, _iterative_round)

    print("Loading state dict from", filename)

    return get_progress_and_state_dict(
        filename,
        _cpu_only
    )['state_dict']


def get_filename(_save_dir: str, _seed: int, _compression_ratio: float, _type: ParameterType,
                 _pruning_type: PruningType, _iterative_round: int):
    return os.path.join(_save_dir,
                        "{}_{}_{:06.3f}_round{:02}_{}.th".format(_pruning_type.value, _seed, _compression_ratio,
                                                                 _iterative_round, _type.value))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return torch.tensor(res)


def print_size(_model):
    parameters_total = 0.0
    parameters_nonzero = 0.0

    for state_key, parameters in _model.named_parameters():
        parameters = parameters.flatten()
        parameters_total += len(parameters)
        parameters_nonzero += (parameters != 0).int().sum()

    print("True size: {} are nonzero. {} in total. {:.2f}% relative size. {:.2f} compression ratio.".format(
        parameters_nonzero,
        parameters_total,
        100 * parameters_nonzero / parameters_total,
        parameters_total / parameters_nonzero
    ))

    return parameters_total / parameters_nonzero


def find_threshold(_cuda, _compression_ratio, _rank_function, initial_state, trained_state):
    ranked_parameters = []

    for state_key in trained_state:
        initial_parameters = _cuda(initial_state[state_key])
        trained_parameters = _cuda(trained_state[state_key])
        ranked_parameters.append(_rank_function(initial_parameters, trained_parameters).view(-1))

    ranked_parameters = torch.cat(ranked_parameters)
    ranked_parameters, _ = torch.sort(ranked_parameters)
    amount_values = len(ranked_parameters)

    size_goal = 1 / _compression_ratio
    threshold_index = int(amount_values * (1 - size_goal))
    threshold = ranked_parameters[threshold_index]

    return threshold.item()


def find_local_threshold(_cuda, _compression_ratio, _rank_function, initial_parameters, trained_parameters):
    ranked_parameters = [_rank_function(initial_parameters, trained_parameters).view(-1)]

    ranked_parameters = torch.cat(ranked_parameters)
    ranked_parameters, _ = torch.sort(ranked_parameters)
    amount_values = len(ranked_parameters)

    size_goal = 1 / _compression_ratio
    threshold_index = int(amount_values * (1 - size_goal))
    threshold = ranked_parameters[threshold_index]

    return threshold.item()


def calculate_previous_compression(_progress):
    ratio = _progress['compression_ratio']
    round = _progress['iterative_round']

    assert round >= 1

    return ratio ** ((round - 1) / round)
