from dataclasses import dataclass
from ga import Grid, CellNN
from ga.cell import ASCIICell
import numpy as np
import torch


@dataclass
class AutomataRunConfiguration:
    gen_rnds: int


def format_ascii_output(output: np.ndarray):
    res = ""
    for row in output:
        for col in row:
            res += chr(col)
        res += "\n"
    return res


def run_automata(cell: CellNN, grid: Grid):  # , run_cfg: AutomataRunConfiguration):
    # grid_shape = (grid.grid_size, grid.grid_size, grid.state_size)
    neighbors = torch.from_numpy(grid.neighbors())
    output = (
        cell(neighbors, output=True).reshape(grid.grid_size, grid.grid_size).numpy()
    )
    print(format_ascii_output(output))


if __name__ == "__main__":
    GRID_SIZE = 4
    STATE_SIZE = 1

    cell = ASCIICell(STATE_SIZE)
    grid = Grid(GRID_SIZE, STATE_SIZE)
    run_automata(cell, grid)
