import torch
import torch.distributed
import torch.distributed.nn
import math


def reduce_any(x, op=torch.distributed.ReduceOp.SUM):
    if torch.distributed.is_initialized():
        is_pyvar = not torch.is_tensor(x)

        if is_pyvar:
            x = torch.tensor(x)
        else:
            device = x.device
            x = x.detach()

        x = x.cuda()
        x = torch.distributed.nn.all_reduce(x, op=op)

        if is_pyvar:
            return x.cpu().item()
        else:
            return x.to(device)
    else:
        return x


def logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False, sync_distributed: bool = True) -> torch.Tensor:
    x = x.float()
    if not (sync_distributed and torch.distributed.is_initialized()):
        return x.logsumexp(dim=dim, keepdim=keepdim)

    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    xmax = torch.distributed.nn.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    xe = torch.distributed.nn.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = (xmax + xe.log())
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0, sync_distributed: bool = True) -> torch.Tensor:
    assert x.shape[dim] > 0
    x = x.float()
    if torch.distributed.is_initialized() and sync_distributed:
        xlse = logsumexp(x, dim=dim)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        n = torch.distributed.nn.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xlse - n.log()
    else:
        return x.logsumexp(dim) - math.log(x.shape[dim])


def mean(x: torch.Tensor, dim: int = 0, sync_distributed: bool = True) -> torch.Tensor:
    assert x.shape[dim] > 0
    x = x.float()
    if torch.distributed.is_initialized() and sync_distributed:
        xs = x.sum(dim=dim)
        xs = torch.distributed.nn.all_reduce(xs, op=torch.distributed.ReduceOp.SUM)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        n = torch.distributed.nn.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xs / n
    else:
        return x.mean(dim=dim)