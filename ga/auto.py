from dataclasses import dataclass
from ga import CellNN, Grid
from typing import Any
import torch
import math


@dataclass
class OptimizerConfig(object):
    optimizer: type
    params: dict[str, Any]


@dataclass
class AutomataConfig(object):
    grid_size: tuple[int, int]
    state_size: int
    output_dim: int
    output_range: int
    optimizer: OptimizerConfig
    gen_rnds: int


class Automata(object):
    def __init__(self, config: AutomataConfig, expected: torch.Tensor):
        self.grid = Grid(*config.grid_size, config.state_size, grid_init_random=True)
        self.cell = CellNN(config.state_size, config.output_dim, config.output_range)
        self.config = config
        self.optimizer: torch.optim.Optimizer = config.optimizer.optimizer(
            self.cell.parameters(), **config.optimizer.params
        )
        self.expected = expected

    def output(self):
        return self.cell.output(self.grid.state()).reshape(
            *self.config.grid_size, self.config.output_dim
        )

    def train(self, train_rnds: int):
        for rnd in range(train_rnds):
            self.reset()
            self._next_state(self.config.gen_rnds)
            loss = self._loss(self.output())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):
        self.reset()
        with torch.no_grad():
            self._next_state(self.config.gen_rnds)
            self._loss(self.output(), is_train=False)

    def reset(self):
        self.grid = Grid(
            *self.config.grid_size, self.config.state_size, grid_init_random=True
        )

    def _next_state(self, rnds: int = 1):
        for rnd in range(rnds):
            neighbors = self.grid.neighbors()
            self.grid = Grid.from_state(self.cell(neighbors), self.config.grid_size)

    def _loss(
        self,
        actual: torch.Tensor,
        is_train: bool = True,
        verbose: bool = True,
    ):
        loss = -torch.sum((self.expected - actual) ** 2) / math.prod(actual.shape)
        if verbose:
            stage = "Train" if is_train else "Validation"
            print(f"Loss ({stage}): {loss.item():.2f}")
        return loss
