from itertools import product
import numpy as np

#################################
# Constants                     #
#################################

NUM_NEIGHBOR_CELLS = 8

#################################
# Grid                          #
#################################


class Grid(object):
    """
    Notation:
        G: Grid Size
        S: State Size
        N: Number of Neighbors
    """

    _NP_DTYPE = np.float64

    def __init__(self, grid_size: int, state_size: int):
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.state_size = state_size
        self._grid = np.zeros((grid_size, grid_size, state_size), dtype=Grid._NP_DTYPE)

    def grid(self) -> np.ndarray:
        """
        Returns grid of shape (G, G, S)
        """

        return self._grid

    def neighbors(self) -> np.ndarray:
        """
        Returns neighbors of each grid cell in a matrix of shape (G*G, N*S)
        """

        # Grid (G, G, S) -> (G*G, S)
        flat_grid = self._grid.reshape(-1, self.state_size)
        assert flat_grid.shape == (self.num_cells, self.state_size)

        # Grab each neighboring position for all cells at once (N, G*G, S)
        sep_neighbors = np.zeros(
            (NUM_NEIGHBOR_CELLS + 1, self.num_cells, self.state_size), Grid._NP_DTYPE
        )

        for ngb_pos, (row_dir, col_dir) in enumerate(
            product(range(-1, 2), range(-1, 2))
        ):
            sep_neighbors[ngb_pos] = self._neighbors_in_dir(flat_grid, row_dir, col_dir)

        return sep_neighbors.transpose(0, 1).reshape(self.num_cells, -1)

    def _neighbors_in_dir(
        self, flat_grid: np.ndarray, row_dir: int, col_dir: int
    ) -> np.ndarray:
        assert abs(row_dir) <= 1 and abs(col_dir) <= 1

        def calc_idx_offset() -> int:
            offset = self.grid_size * row_dir
            offset += self.grid_size + col_dir
            return offset

        idx_offset = calc_idx_offset()

        neighbors_in_dir = np.zeros((self.num_cells, self.state_size), Grid._NP_DTYPE)

        for cell_idx in range(self.num_cells):
            ngb_idx = cell_idx + idx_offset

            if self._is_flat_idx_valid(ngb_idx):
                neighbors_in_dir[cell_idx] = flat_grid[ngb_idx]

        return neighbors_in_dir

    def _flat_idx_to_coord(self, flat_idx: int) -> tuple[int, int]:
        assert self._is_flat_idx_valid(flat_idx)
        return flat_idx // self.grid_size, flat_idx % self.grid_size

    def _coord_to_flat_idx(self, coord: tuple[int, int]) -> int:
        assert self._is_coord_valid(coord)
        return coord[0] * self.grid_size + coord[1]

    def _is_coord_valid(self, coord: tuple[int, int]) -> bool:
        return (
            0 <= coord[0]
            and coord[0] < self.grid_size
            and 0 <= coord[1]
            and coord[1] < self.grid_size
        )

    def _is_flat_idx_valid(self, flat_idx: int) -> bool:
        return 0 <= flat_idx and flat_idx < self.num_cells
