import time

import torch.nn.parallel
import torch.optim
import torch.utils.data

from imports.utils import AverageMeter, compute_accuracy, print_size


def validate(_val_loader, _model, _criterion, _cuda, _args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    _model.eval()

    print_size(_model)

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(_val_loader):
            target = _cuda(target)
            input_var = _cuda(input)
            target_var = _cuda(target)

            if _args.half:
                input_var = input_var.half()

            # compute output
            output = _model(input_var)
            loss = _criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            _accuracy = compute_accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            accuracies.update(_accuracy, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % _args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {accuracies.val} ({accuracies.avg})'
                        .format(i, len(_val_loader), batch_time=batch_time, loss=losses, accuracies=accuracies)
                )

    print(' * Prec@1 {accuracies.avg}'.format(accuracies=accuracies))

    return accuracies.avg.tolist()
