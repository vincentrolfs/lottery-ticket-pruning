from collections import OrderedDict

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from imports.utils import ParameterType, get_state_dict
from imports.utils import PruningType, calculate_previous_compression, find_threshold

rank_function = lambda initial_parameters, trained_parameters: torch.abs(trained_parameters)
reset_absolute_value_function = lambda initial_parameters, trained_parameters: torch.abs(initial_parameters)
reset_sign_function = lambda initial_parameters, trained_parameters: torch.sign(initial_parameters)


def prune(_args, _cuda, _model, _progress):
    prev_compression = calculate_previous_compression(_progress)
    initial_state = get_state_dict(_args.save_dir, _progress['seed'], prev_compression,
                                   ParameterType.INITIAL, _args.cpu_only, PruningType.LT_TRADITIONAL,
                                   _progress['iterative_round'] - 1)
    trained_state = get_state_dict(_args.save_dir, _progress['seed'], prev_compression,
                                   ParameterType.FINAL, _args.cpu_only, PruningType.LT_TRADITIONAL,
                                   _progress['iterative_round'] - 1)
    threshold = find_threshold(_cuda, _progress['compression_ratio'], rank_function, initial_state, trained_state)

    new_state = OrderedDict()

    parameters_total = 0.0
    parameters_kept = 0.0

    _all_masks = {}

    for state_key in trained_state:
        initial_parameters = _cuda(initial_state[state_key])
        trained_parameters = _cuda(trained_state[state_key])

        mask = rank_function(initial_parameters, trained_parameters) >= threshold
        _all_masks[state_key] = mask.clone()

        absolute_value_reset = reset_absolute_value_function(initial_parameters, trained_parameters)
        sign_reset = reset_sign_function(initial_parameters, trained_parameters)
        reset = sign_reset * absolute_value_reset

        new_parameters = torch.zeros_like(initial_parameters)
        new_parameters[mask] = reset[mask]
        new_state[state_key] = new_parameters

        mask_flat = mask.flatten()
        parameters_total += len(mask_flat)
        parameters_kept += mask_flat.sum().item()

    _model.load_state_dict(new_state)

    print(parameters_kept, parameters_total)

    return parameters_kept / parameters_total, _all_masks
