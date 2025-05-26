from itertools import product
from typing import Optional
from time import time
import torch

#################################
# Constants                     #
#################################

NUM_NEIGHBOR_CELLS = 8

#################################
# Coordinate                    #
#################################


class Coordinate(object):
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __add__(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.row + other.row, self.col + other.col)

    def __iadd__(self, other: "Coordinate"):
        self.row += other.row
        self.col += other.col

    def __eq__(self, other) -> bool:
        if not isinstance(other, Coordinate):
            return False
        return self.row == other.row and self.col == other.col

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"

    def __repr__(self) -> str:
        return str(self)


#################################
# Rectangle                     #
#################################


class Rectangle(object):
    def __init__(self, top_left: Coordinate, shape: tuple[int, int]):
        self.top_left = top_left
        self.shape = shape
        self.top, self.left = top_left.row, top_left.col
        self.rows, self.cols = shape

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return (
            self.top_left == other.top_left
            and self.rows == other.rows
            and self.cols == other.cols
        )

    def __str__(self) -> str:
        return f"({self.top}, {self.left}, {self.rows}, {self.cols})"

    def __repr__(self) -> str:
        return str(self)


#################################
# Grid                          #
#################################


class Grid(object):
    """
    Notation:
        G_R: Grid Rows
        G_C: Grid Cols
        G: Number of cells (G_R * G_C)
        S: State Size
        N: Number of Neighbors
    """

    _TORCH_DTYPE = torch.float32

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        state_size: int,
        grid: Optional[torch.Tensor] = None,
        grid_init_random: bool = False,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_size = (grid_rows, grid_cols)
        self.num_cells = grid_rows * grid_cols
        self.state_size = state_size

        if grid is None and not grid_init_random:
            self._grid = torch.zeros(
                (grid_rows, grid_cols, state_size), dtype=Grid._TORCH_DTYPE
            )
        elif grid is None and grid_init_random:
            self._grid = torch.rand(
                (grid_rows, grid_cols, state_size), dtype=Grid._TORCH_DTYPE
            )
        else:
            assert grid is not None
            assert grid.shape == (grid_rows, grid_cols, state_size)
            assert grid.dtype == Grid._TORCH_DTYPE
            self._grid = grid

    @classmethod
    def from_grid(cls, grid: torch.Tensor):
        grid_rows, grid_cols, state_size = grid.shape
        return cls(grid_rows, grid_cols, state_size, grid=grid)

    @classmethod
    def from_state(cls, state: torch.Tensor, grid_size: tuple[int, int]):
        return cls.from_grid(state.reshape(*grid_size, -1))

    def grid(self) -> torch.Tensor:
        """
        Returns grid of shape (G_R, G_C, S)
        """

        return self._grid

    def state(self) -> torch.Tensor:
        """
        Returns state of each grid cell in a matrix of shape (G, S)
        """

        return self._grid.reshape(-1, self.state_size)

    def neighbors(self) -> torch.Tensor:
        """
        Returns neighbors of each grid cell in a matrix of shape (G, N*S)
        """

        # Grab each neighboring position for all cells at once (N, G*G, S)
        sep_neighbors = torch.zeros(
            (NUM_NEIGHBOR_CELLS + 1, self.num_cells, self.state_size),
            dtype=Grid._TORCH_DTYPE,
        )

        for ngb_pos, direction in enumerate(product(range(-1, 2), range(-1, 2))):
            sep_neighbors[ngb_pos] = self._neighbors_in_dir(Coordinate(*direction))

        return sep_neighbors.transpose(0, 1).reshape(self.num_cells, -1)

    def _neighbors_in_dir(self, direction: Coordinate) -> torch.Tensor:
        """
        Returns neighbors in a single direction (G, S)
        """

        assert abs(direction.row) <= 1 and abs(direction.col) <= 1

        # (G_R, G_C, S)
        neighbors_in_dir = torch.zeros(
            (*self.grid_size, self.state_size), dtype=Grid._TORCH_DTYPE
        )

        def cpy_bnds(
            uni_dir: int, length: int
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            if uni_dir == -1:
                return (0, length - 1), (1, length)
            elif uni_dir == 0:
                return (0, length), (0, length)
            else:
                return (1, length), (0, length - 1)

        src_vert_bnds, tgt_vert_bnds = cpy_bnds(direction.row, self.grid_rows)
        src_horz_bnds, tgt_horz_bnds = cpy_bnds(direction.col, self.grid_cols)

        neighbors_in_dir[
            tgt_vert_bnds[0] : tgt_vert_bnds[1], tgt_horz_bnds[0] : tgt_horz_bnds[1]
        ] = self._grid[
            src_vert_bnds[0] : src_vert_bnds[1], src_horz_bnds[0] : src_horz_bnds[1]
        ]

        return neighbors_in_dir.reshape(self.num_cells, self.state_size)

    def _flat_idx_to_coord(self, flat_idx: int) -> Coordinate:
        assert self._is_flat_idx_valid(flat_idx)
        return Coordinate(flat_idx // self.grid_cols, flat_idx % self.grid_cols)

    def _coord_to_flat_idx(self, coord: Coordinate) -> int:
        assert self._is_coord_valid(coord)
        return coord.row * self.grid_cols + coord.col

    def _is_coord_valid(self, coord: Coordinate) -> bool:
        return (
            0 <= coord.row
            and coord.row < self.grid_rows
            and 0 <= coord.col
            and coord.col < self.grid_cols
        )

    def _is_flat_idx_valid(self, flat_idx: int) -> bool:
        return 0 <= flat_idx and flat_idx < self.num_cells

    def _get_cell(self, coord: Coordinate) -> torch.Tensor:
        assert self._is_coord_valid(coord)
        return self._grid[coord.row][coord.col]
