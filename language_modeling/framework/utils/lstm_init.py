import torch
import torch.nn
from typing import Union
import math


def lstm_init_forget(lstm: Union[torch.nn.LSTM, torch.nn.LSTMCell], bias: float = 1):
    if isinstance(lstm, torch.nn.LSTM):
        biases = [k for k in dir(lstm) if k.startswith("bias_")]
    elif isinstance(lstm, torch.nn.LSTMCell):
        biases = ["bias_hh", "bias_ih"]
    else:
        assert False, "Invalid lstm type"

    for b in biases:
        var = lstm.__getattr__(b)
        if b.startswith("bias_hh"):
            state_size = var.shape[0] // 4
            var.data[state_size:2*state_size] = bias


def lstm_init(lstm: torch.nn.LSTM):
    for name in dir(lstm):
        if name.startswith("bias_hh"):
            # Forget gate bias to 1, rest to 0
            var = lstm.__getattr__(name)
            state_size = var.shape[0] // 4
            var.data.fill_(0)
            var.data[state_size:2 * state_size] = 1
        elif name.startswith("bias_ih"):
            # Redundant input biases to 0
            var = lstm.__getattr__(name)
            var.data.fill_(0)
        elif name.startswith("weight_"):
            # Xavier init for most transforms, except orthogonal for the hidden to data
            var = lstm.__getattr__(name)
            state_size = var.shape[0]//4
            std = math.sqrt(2 / (state_size + var.shape[1])) / math.sqrt(2) * \
                  torch.nn.init.calculate_gain("sigmoid")

            var.data.normal_(0, std)
            if name.startswith("weight_hh_"):
                torch.nn.init.orthogonal_(var[2 * state_size:3 * state_size].data,
                                          gain=torch.nn.init.calculate_gain("tanh") / math.sqrt(2))
            else:
                var[2 * state_size:3 * state_size].data *= torch.nn.init.calculate_gain("tanh") / \
                                                           torch.nn.init.calculate_gain("sigmoid")
