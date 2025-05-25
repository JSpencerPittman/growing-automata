import numpy as np
import pytest

from ga import Grid, Coordinate


@pytest.fixture
def grid_8by8_zeroed():
    return Grid(8, 7)


@pytest.fixture
def grid_3by3_enumerated():
    grid = Grid(3, 1)
    grid._grid = np.arange(1, 10, dtype=Grid._NP_DTYPE).reshape(3, 3, 1)
    return grid


@pytest.fixture
def grid_2by2_enumerated():
    grid = Grid(2, 2)
    grid._grid = np.arange(1, 9, dtype=Grid._NP_DTYPE).reshape(2, 2, 2)
    return grid


def are_numpy_arrays_the_same(a: np.ndarray, b: np.ndarray):
    return a.dtype == b.dtype and a.shape == b.shape and (a - b).sum() == 0.0


def test_Grid__init(grid_8by8_zeroed):
    assert grid_8by8_zeroed.num_cells == 64
    assert grid_8by8_zeroed._grid.shape == (8, 8, 7)


def test_Grid__neighbors(grid_3by3_enumerated):
    """
    Test neighbors on 3x3 simple grid
    """

    exp_neighbors = np.array(
        [
            [0, 0, 0, 0, 1, 2, 0, 4, 5],
            [0, 0, 0, 1, 2, 3, 4, 5, 6],
            [0, 0, 0, 2, 3, 0, 5, 6, 0],
            [0, 1, 2, 0, 4, 5, 0, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 3, 0, 5, 6, 0, 8, 9, 0],
            [0, 4, 5, 0, 7, 8, 0, 0, 0],
            [4, 5, 6, 7, 8, 9, 0, 0, 0],
            [5, 6, 0, 8, 9, 0, 0, 0, 0],
        ],
        dtype=Grid._NP_DTYPE,
    )

    act_neighbors = grid_3by3_enumerated.neighbors()

    assert are_numpy_arrays_the_same(exp_neighbors, act_neighbors)


def test_Grid__prv_neighbors_in_dir(grid_3by3_enumerated, grid_2by2_enumerated):
    """
    Test neighbors on 3x3 simple grid
    """

    top_left_dir = Coordinate(-1, -1)
    exp_neighbor_in_top_left = np.array(
        [0, 0, 0, 0, 1, 2, 0, 4, 5], dtype=Grid._NP_DTYPE
    ).reshape(9, 1)
    act_neighbor_in_top_left = grid_3by3_enumerated._neighbors_in_dir(top_left_dir)

    assert are_numpy_arrays_the_same(exp_neighbor_in_top_left, act_neighbor_in_top_left)

    """
    Test neighbors on 2x2 grid with state size of 3
    """

    bot_right_dir = Coordinate(1, 1)
    exp_neighbor_in_bot_right = np.array(
        [7, 8, 0, 0, 0, 0, 0, 0], dtype=Grid._NP_DTYPE
    ).reshape(4, 2)
    act_neighbor_in_bot_right = grid_2by2_enumerated._neighbors_in_dir(bot_right_dir)

    assert are_numpy_arrays_the_same(
        exp_neighbor_in_bot_right, act_neighbor_in_bot_right
    )


def test_Grid___prv_flat_idx_to_coord(grid_8by8_zeroed):
    assert grid_8by8_zeroed._flat_idx_to_coord(0) == Coordinate(0, 0)
    assert grid_8by8_zeroed._flat_idx_to_coord(10) == Coordinate(1, 2)
    assert grid_8by8_zeroed._flat_idx_to_coord(63) == Coordinate(7, 7)


def test_Grid___prv_coord_to_flat_idx(grid_8by8_zeroed):
    assert grid_8by8_zeroed._coord_to_flat_idx(Coordinate(0, 0)) == 0
    assert grid_8by8_zeroed._coord_to_flat_idx(Coordinate(1, 2)) == 10
    assert grid_8by8_zeroed._coord_to_flat_idx(Coordinate(7, 7)) == 63


def test_Grid__prv_is_coord_valid(grid_8by8_zeroed):
    assert not grid_8by8_zeroed._is_coord_valid(Coordinate(-1, 0))
    assert not grid_8by8_zeroed._is_coord_valid(Coordinate(8, 3))
    assert not grid_8by8_zeroed._is_coord_valid(Coordinate(0, -1))
    assert not grid_8by8_zeroed._is_coord_valid(Coordinate(3, 8))
    assert grid_8by8_zeroed._is_coord_valid(Coordinate(0, 0))
    assert grid_8by8_zeroed._is_coord_valid(Coordinate(7, 7))


def test_Grid__prv_is_flat_idx_valid(grid_8by8_zeroed):
    assert not grid_8by8_zeroed._is_flat_idx_valid(-1)
    assert grid_8by8_zeroed._is_flat_idx_valid(0)
    assert grid_8by8_zeroed._is_flat_idx_valid(63)
    assert not grid_8by8_zeroed._is_flat_idx_valid(64)
