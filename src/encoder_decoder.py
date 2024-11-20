from datasets import load_from_disk
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Args:
            latent_dim: Dimension of latent space (encoded vector)
            hidden_dims: List of hidden layer dimensions (in reverse order of encoder)
            output_dim: Dimension of output vector
        """
        super().__init__()

        # Build decoder layers
        layers = []
        prev_dim = latent_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Add final layer to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class VectorEncoderDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, output_dim: int):
        """
        Args:
            input_dim: Dimension of input vector
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space (encoded vector)
            output_dim: Dimension of output vector
        """
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded