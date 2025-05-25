from itertools import product
import numpy as np

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

        # Grab each neighboring position for all cells at once (N, G*G, S)
        sep_neighbors = np.zeros(
            (NUM_NEIGHBOR_CELLS + 1, self.num_cells, self.state_size), Grid._NP_DTYPE
        )

        for ngb_pos, direction in enumerate(product(range(-1, 2), range(-1, 2))):
            sep_neighbors[ngb_pos] = self._neighbors_in_dir(Coordinate(*direction))

        return sep_neighbors.transpose(1, 2, 0).reshape(self.num_cells, -1)

    def _neighbors_in_dir(self, direction: Coordinate) -> np.ndarray:
        """
        Returns neighbors in a single direction (G*G, S)
        """

        assert abs(direction.row) <= 1 and abs(direction.col) <= 1

        neighbors_in_dir = np.zeros((self.num_cells, self.state_size), Grid._NP_DTYPE)

        for cell_idx in range(self.num_cells):
            ngb_coord = self._flat_idx_to_coord(cell_idx) + direction

            if self._is_coord_valid(ngb_coord):
                neighbors_in_dir[cell_idx] = self._get_cell(ngb_coord)

        return neighbors_in_dir

    def _flat_idx_to_coord(self, flat_idx: int) -> Coordinate:
        assert self._is_flat_idx_valid(flat_idx)
        return Coordinate(flat_idx // self.grid_size, flat_idx % self.grid_size)

    def _coord_to_flat_idx(self, coord: Coordinate) -> int:
        assert self._is_coord_valid(coord)
        return coord.row * self.grid_size + coord.col

    def _is_coord_valid(self, coord: Coordinate) -> bool:
        return (
            0 <= coord.row
            and coord.row < self.grid_size
            and 0 <= coord.col
            and coord.col < self.grid_size
        )

    def _is_flat_idx_valid(self, flat_idx: int) -> bool:
        return 0 <= flat_idx and flat_idx < self.num_cells

    def _get_cell(self, coord: Coordinate) -> np.ndarray:
        assert self._is_coord_valid(coord)
        return self._grid[coord.row][coord.col]
