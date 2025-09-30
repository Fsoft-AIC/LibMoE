import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# FiLM Generator (produces gamma and beta)
class FiLMGenerator(nn.Module):
    def __init__(self, input_dim, num_features, output_dim):
        """
        Generate gamma and beta from conditioning input.
        :param input_dim: Dimension of the conditioning input (e.g., question embedding).
        :param num_features: Number of feature maps to modulate.
        """
        super(FiLMGenerator, self).__init__()

        self.htoh4 = nn.Linear(input_dim, num_features // 2, bias=True)
        self.h4toh = nn.Linear(num_features // 2, output_dim * 2, bias=True)
        self.activation = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize htoh4 weights and bias
        nn.init.kaiming_uniform_(self.htoh4.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.htoh4.bias)

        # Initialize h4toh weights and biases
        nn.init.zeros_(self.h4toh.weight)  # Zero out weights to ensure gamma and beta start with specific values
        nn.init.zeros_(self.h4toh.bias)

        # Modify biases to produce gamma=1 and beta=0
        d_model = self.h4toh.out_features // 2
        self.h4toh.bias.data[:d_model] = 1.0  # Gamma initialized to 1
        self.h4toh.bias.data[d_model:] = 0.0  # Beta initialized to 0

    def forward(self, x):
        """
        Generate gamma and beta.
        :param x: Conditioning input (batch_size, input_dim).
        """
        params = self.htoh4(x)
        params = self.activation(params)
        params = self.h4toh(params) # (batch_size, 2 * num_features)
        gamma, beta = params.chunk(2, dim=-1)  # Split into gamma and beta
        return gamma, beta


if __name__=='__main__':
    film_layer = FiLMGenerator(input_dim=8, num_features=24, output_dim=8)