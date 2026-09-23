import numpy as np
import pytest

from kite.operators import cell_mean, floored, solve, tilt, tilt_cell

VALUES = np.arange(1, 7, dtype=float)  # a 1-6 rating scale


def cell(seed: int = 0, rows: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(6, 0.7), size=rows)
    return np.round(p, 2) / np.round(p, 2).sum(axis=1, keepdims=True)  # two-decimal rounding, as the kernel reports


def test_zero_alpha_is_the_floored_input():
    p = cell()
    assert np.allclose(tilt(floored(p), VALUES, 0.0), floored(p))
    assert np.allclose(floored(p).sum(axis=1), 1.0)
    assert (floored(p) > 0).all()
    assert floored(p).min() >= 0.005 / (1 + 6 * 0.005)  # a floored entry shrinks only by the renormalisation


def test_mean_is_increasing_in_alpha():
    p = floored(cell(1))
    means = [cell_mean(tilt(p, VALUES, a), VALUES) for a in np.linspace(-6, 6, 25)]
    assert (np.diff(means) > 0).all()


@pytest.mark.parametrize("shift", [-0.6, -0.1, 0.05, 0.4])
def test_solve_reaches_the_target(shift):
    p = cell(2)
    target = cell_mean(floored(p), VALUES) + shift
    tilted, result = tilt_cell(p, VALUES, target)
    assert result.feasible
    assert abs(result.achieved - target) < 1e-6
    assert abs(cell_mean(tilted, VALUES) - target) < 1e-6
    assert np.allclose(tilted.sum(axis=1), 1.0)
    assert (tilted > 0).all()


def test_weights_define_the_cell_mean():
    p = cell(3)
    w = np.zeros(len(p))
    w[:5] = 1.0  # only five rows count
    target = cell_mean(floored(p)[:5], VALUES) + 0.3
    tilted, result = tilt_cell(p, VALUES, target, w)
    assert result.feasible
    assert abs(cell_mean(tilted[:5], VALUES) - target) < 1e-6


def test_infeasible_target_leaves_the_cell_alone():
    p = cell(4)
    baseline = cell_mean(floored(p), VALUES)
    tilted, result = tilt_cell(p, VALUES, 6.0)  # the top of the scale is not reachable within the alpha cap
    assert not result.feasible
    assert result.requested == 6.0
    assert abs(result.achieved - baseline) < 1e-12
    assert result.lower < baseline < result.upper < 6.0
    assert np.allclose(tilted, floored(p))


def test_alpha_has_the_same_meaning_on_any_scale():
    p = floored(cell(5))
    on_ratings = cell_mean(tilt(p, VALUES, 2.0), VALUES)
    on_unit = cell_mean(tilt(p, (VALUES - 1) / 5, 2.0), (VALUES - 1) / 5)
    assert abs((on_ratings - 1) / 5 - on_unit) < 1e-12


def test_tilt_is_undone_by_minus_alpha():
    p = floored(cell(6))
    assert np.allclose(tilt(tilt(p, VALUES, 1.7), VALUES, -1.7), p)


def test_floor_bounds_what_a_row_can_reach():
    p = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    no_floor = solve(p, VALUES, 5.5, floor=0.0)
    assert not no_floor.feasible and no_floor.upper <= 2.0 + 1e-9
    with_floor = solve(p, VALUES, 5.5, floor=0.005)
    assert with_floor.feasible


def test_constant_values_are_rejected():
    with pytest.raises(ValueError):
        tilt(floored(cell()), np.ones(6), 1.0)
