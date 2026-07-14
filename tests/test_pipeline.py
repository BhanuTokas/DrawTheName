import numpy as np

from drawthename.pipeline import _stratified_subsample


def test_stratified_subsample_noop_when_under_budget():
    groups = [np.zeros((10, 4)), np.zeros((5, 4))]
    result = _stratified_subsample(groups, max_total=100, rng=np.random.default_rng(0))
    assert result is groups


def test_stratified_subsample_preserves_proportions_and_budget():
    rng = np.random.default_rng(0)
    groups = [np.zeros((1000, 4)), np.zeros((100, 4)), np.zeros((10, 4))]
    result = _stratified_subsample(groups, max_total=222, rng=rng)

    sizes = [len(g) for g in result]
    # every non-empty group keeps at least one point
    assert all(n >= 1 for n in sizes)
    # roughly matches the input proportions (1000:100:10 ~= 10:1:0.1)
    assert sizes[0] > sizes[1] > sizes[2]
    # total stays close to the budget (small overage allowed from the min-1 floor)
    assert sum(sizes) <= 222 + len(groups)


def test_stratified_subsample_never_exceeds_group_size():
    rng = np.random.default_rng(0)
    groups = [np.zeros((3, 4)), np.zeros((3, 4))]
    result = _stratified_subsample(groups, max_total=4, rng=rng)
    assert all(len(g) <= 3 for g in result)
