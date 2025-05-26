import torch.nn.functional as F
from torch import nn
import torch

#################################
# Constants                     #
#################################

NUM_NEIGHBOR_CELLS = 8

#################################
# CellNN                        #
#################################


class CellNN(nn.Module):
    _TORCH_DTYPE = torch.float32
    _TORCH_OUTPUT_DTYPE = torch.int32

    def __init__(self, state_size: int, output_dim: int, num_poss_vals: int):
        super().__init__()

        self.in_dim = (NUM_NEIGHBOR_CELLS + 1) * state_size
        self.out_dim = output_dim
        self.num_poss_vals = num_poss_vals
        self.state_size = state_size
        self.map_state = nn.Linear(
            self.in_dim, state_size, bias=True, dtype=CellNN._TORCH_DTYPE
        )
        self.map_output = nn.Linear(
            state_size, self.out_dim, bias=True, dtype=CellNN._TORCH_DTYPE
        )

    def next_state(self, ngbs_state: torch.Tensor) -> torch.Tensor:
        """
        Map the state to the next state.

        Args:
            ngbs_state (torch.Tensor): Neighbor's state (G*G, N*S)

        Returns:
            torch.Tensor: Next state (G*G, S)
        """

        assert ngbs_state.ndim == 2
        assert ngbs_state.dtype == CellNN._TORCH_DTYPE
        assert ngbs_state.size(1) == (NUM_NEIGHBOR_CELLS + 1) * self.state_size

        return self.map_state(ngbs_state)

    def output(self, state: torch.Tensor) -> torch.Tensor:
        """
        Map the state to the output.

        Args:
            state (torch.Tensor): State (G*G, S)

        Returns:
            torch.Tensor: Output (G*G, O)
        """

        assert state.ndim == 2
        assert state.dtype == CellNN._TORCH_DTYPE
        assert state.size(1) == self.state_size

        return F.sigmoid(self.map_output(state)) * self.num_poss_vals

    def forward(self, curr_state: torch.Tensor, output: bool = False) -> torch.Tensor:
        x = self.next_state(curr_state)
        if output:
            x = self.output(x)
        return x


class ASCIICell(CellNN):
    _ASCII_FRST_CHR = ord(" ")
    _ASCII_LAST_CHR = ord("~")
    _ASCII_NUM_CHRS = _ASCII_LAST_CHR - _ASCII_FRST_CHR + 1

    def __init__(self, state_size):
        super().__init__(state_size, 1, ASCIICell._ASCII_NUM_CHRS)

    def output(self, state: torch.Tensor, fmt: bool = False) -> torch.Tensor:
        return super().output(state) + ASCIICell._ASCII_FRST_CHR
