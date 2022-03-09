import time

import torch.nn.parallel
import torch.optim
import torch.utils.data

from imports.utils import AverageMeter, compute_accuracy


def apply_freezing(_model, _freeze_masks):
    if _freeze_masks is None:
        return

    for state_key, parameters in _model.named_parameters():
        parameters.grad = torch.mul(parameters.grad, _freeze_masks[state_key])


def train(_train_loader, _model, _criterion, _optimizer, _epoch, _cuda, _args, _freeze_masks=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    _model.train()

    end = time.time()
    for i, (input, target) in enumerate(_train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = _cuda(target)
        input_var = _cuda(input)
        target_var = target
        if _args.half:
            input_var = input_var.half()

        # compute output
        output = _model(input_var)
        loss = _criterion(output, target_var)

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        apply_freezing(_model, _freeze_masks)
        _optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        _accuracy = compute_accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        accuracies.update(_accuracy.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % _args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {accuracies.val:.3f} ({accuracies.avg:.3f})'.format(
                _epoch, i, len(_train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, accuracies=accuracies))
