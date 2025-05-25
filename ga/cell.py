from torch import nn
import torch

#################################
# Constants                     #
#################################

NUM_NEIGHBOR_CELLS = 8
NUM_COLOR_CHANNELS = 3

#################################
# CellNN                        #
#################################


class CellNN(object):
    def __init__(self, state_size: int):
        self.in_dim = (NUM_NEIGHBOR_CELLS + 1) * state_size
        self.out_dim = NUM_COLOR_CHANNELS
        self.state_size = state_size
        self.map_state = nn.Linear(self.in_dim, state_size, bias=True)
        self.map_output = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def next_state(self, curr_state: torch.Tensor) -> torch.Tensor:
        """
        Map the state to the next state.

        Args:
            curr_state (torch.Tensor): Current state (G*G, N*S)

        Returns:
            torch.Tensor: Next state (G*G, S)
        """

        assert curr_state.ndim == 2
        assert curr_state.size(1) == (NUM_NEIGHBOR_CELLS + 1) * self.state_size

        return self.map_state(curr_state)

    def output(self, state: torch.Tensor) -> torch.Tensor:
        """
        Map the state to the output.

        Args:
            state (torch.Tensor): State (G*G, S)

        Returns:
            torch.Tensor: Output (G*G, O)
        """

        assert state.ndim == 2
        assert state.size(1) == self.state_size

        return self.map_output(state)
