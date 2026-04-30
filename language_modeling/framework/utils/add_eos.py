import torch


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int, batch_dim: int = 1, pad_val=0) -> torch.Tensor:
    time_dim = 1 - batch_dim
    input = torch.cat((input, torch.full_like(input.select(time_dim, 0).unsqueeze(time_dim), pad_val)), dim=time_dim)
    input.scatter_(time_dim, lengths.unsqueeze(time_dim).long(), value=eos_id)
    return input
