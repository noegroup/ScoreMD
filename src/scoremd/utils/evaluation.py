import jax
import jax.numpy as jnp
from scipy.spatial.distance import jensenshannon
from typing import Union, List, Tuple, Callable
import pandas as pd
import numpy as np


def js_divergence(truth, samples, bins=100, limits=None):
    # Determine bin edges based on combined data range
    if limits is None:
        all_data = jnp.vstack([truth, samples])
        x_edges = jnp.linspace(all_data[:, 0].min(), all_data[:, 0].max(), bins + 1)
        y_edges = jnp.linspace(all_data[:, 1].min(), all_data[:, 1].max(), bins + 1)
    else:
        (x_min, x_max), (y_min, y_max) = limits
        x_edges = jnp.linspace(x_min, x_max, bins + 1)
        y_edges = jnp.linspace(y_min, y_max, bins + 1)

    # Compute 2D histograms (joint probability distributions)
    P, _, _ = jnp.histogram2d(truth[:, 0], truth[:, 1], bins=[x_edges, y_edges], density=True)
    Q, _, _ = jnp.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges], density=True)

    # Flatten the histograms to turn them into 1D probability distributions
    P = P.flatten()
    Q = Q.flatten()

    # Compute the Jensen-Shannon distance
    js_distance = jensenshannon(P, Q)

    # Square to get Jensen-Shannon divergence
    return js_distance**2


@jax.vmap
def pairwise_distances(conformation: jnp.ndarray) -> jnp.ndarray:
    """Compute the parwise distances of a batch of conformations."""
    # compute the l2 distance between all atom pairs
    conformation = conformation.reshape(-1, 3)  # has shape (num_atoms, 3)
    return jnp.linalg.norm(conformation[:, None, :] - conformation[None, :, :], axis=-1)


def sgridify(table: pd.DataFrame, columns: List[str], limits: List[Tuple[float, float]], **kwargs) -> pd.Series:
    """Assign bins and computes histogram across various columns.

    Arguments:
    ---------
    table (pandas.DataFrame):
        Table containing data for discretization. May contain more than just that data.
    columns (list of strings):
        Name of columns that will be individually discretized.
    limits (list of lists of 2 floats):
        Each element specifies the max and min of the grid used for each axis we
        discretize. See n_bins for more details. Should have the same length
        as columns
    kwargs:
        Passed to discretize

    Returns:
    -------
    Series with a pandas.MultiIndex specifying the ids of the bins
    and the values specifying the proportion of samples in that bin.

    """
    n_samples = len(table)
    results = []
    for col, (low, high) in zip(columns, limits):
        data = table.loc[:, col].to_numpy()
        disc, labels = discretize(data, low=low, high=high, **kwargs)
        results.append(labels[disc])
    tab = pd.DataFrame(zip(*results), columns=columns)
    pr = tab.groupby(columns).size() / n_samples
    return pr  # type: ignore # I think mypy is wrong here? not completely sure


def fe_sq_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Square error through -log."""
    fex = -np.log(x)
    fey = -np.log(y)
    return (fex - fey) ** 2


def sq_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Square error (x-y)**2. Applies to numpy arrays."""
    return (x - y) ** 2


def mjs_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jensen Shannon error.

    If averaged over the mixture distribution, the expectation of this error is the
    Jensen Shannon divergence. Assumes x and y are probability densities.
    """
    m = (x + y) / 2
    tx = x / m * np.log(x / m)
    ty = y / m * np.log(y / m)
    return (tx + ty) / 2


def discretize(
    data: np.ndarray,
    low: float,
    high: float,
    n_bins: int,
    outlier_check: bool,
    labels: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate a set of bins and assigns each element in an array to a bin.

    Arguments:
    ---------
    data (numpy.ndarray):
        Data to discretize
    low (float):
        Lower "limit" of grid; see outlier_check.
    high (float):
        Upper "limit" of grid; see outlier_check.
    n_bins (positive integer):
        Number of bins to use when discretizing each feature. Note that two of
        these bins are the outlier bins that go outside the proposed limits. See
        outlier_check.
    outlier_check (boolean):
        The created grid has a bin which contains all points lower than "low" and
        another for points higher than "high". If outlier_check is true, if any
        points occupy these special outer bins, a value is raised. If not, the bins
        still exist and aggregate counts but no exception is raised.
    labels (boolean):
        If true, then we return an ordered pair, where the first element is the binned
        data and the second elements is labels; if false, just the binned data.

    NOTE: Only n_bins-2 individual bins are between low and high. See outlier_check.

    Returns:
    -------
    See labels.

    """
    grid = np.linspace(start=low, stop=high, num=(n_bins - 1), endpoint=True)
    binned = np.digitize(data, grid)
    if outlier_check:
        if np.any(binned == 0) or np.any(binned == (n_bins - 1)):
            raise ValueError("Points occupy outside bins. Stopping.")

    if labels:
        labs = np.full(len(grid) + 1, np.nan)
        labs[1:-1] = grid[0:-1] - (np.diff(grid) / 2)
        labs[0] = -np.inf
        labs[-1] = np.inf
        return (binned, labs)
    else:
        return binned


def grid_pointwise(
    left: pd.DataFrame,
    right: pd.DataFrame,
    n_bins: int,
    limits: List[Tuple[float, float]],
    weight: str = "outer",
    baseline: float = 0.0,
    loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = sq_error,
    return_table: bool = False,
    density: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, float]:
    """Compare two tables using grids on a set of their columns.

    Arguments:
    ---------
    left (pandas.DataFrame):
        Table with the first dataset to compare.
    right (pandas.DataFrame):
        Table with the second dataset to compare.
    n_bins (positive integer):
        Number of bins to use when discretizing each feature. Note that two of
        these bins are the outlier bins that go outside the proposed limits.
    limits (list of lists of 2 floats):
        Each element specifies the max and min of the grid used for each axis we
        discretize. See n_bins for more details.
    weight (string):
        Strategy used for weighting the bin errors when taking their average.
        Can be one of the following:
            "outer":
                Uniformly weighted across bins occupied by either frames.
            "inner":
                Uniformly weighted across bins occupied by both frames.
            "left":
                Weighted by the probabilities from the left frame.
            "right":
                Weighted by the probabilities from the right frame.
            "mix":
                Weighted by the average of the left and right probabilities
            "uleft":
                Uniform left. Only include errors on bins occupied by the left
                frame.
            "uright":
                Uniform right. Only include errors on bins occupied by the right
                frame.
    baseline (float):
        When weight specifies bins where one frame may not have any values,
        baseline is used to fill those bins. 0 is a common option, but if free
        energies are taken it may have to be instead set to a small value.
    loss (callable):
        Vectorized callable that that takes the left probabilities and right
        probabilities as input and outputs the error associated with each bin.
        See sq_error as and example.
    return_table (boolean):
        Instead of returning a scalar score, return the aggregate DataFrame
        used to calculate that score.
    density (boolean):
        If true, the size of each grid block is computed and the count number
        for that bin is divided by this size. This new value is assigned to
        the bin and used in loss calculations. Note that the original
        non-divided probabilities are used for weighting when calculating
        the final score.
    kwargs:
        passed to sgridify. Likely you will have to pass columns and
        outlier_check.

    Returns:
    -------
    If return_table: returns a aggregate table containing the errors and bins.
    Else, returns a scalar.

    """
    left_grid = sgridify(left, limits=limits, n_bins=n_bins, **kwargs)
    right_grid = sgridify(right, limits=limits, n_bins=n_bins, **kwargs)
    agg = pd.DataFrame({"left": left_grid, "right": right_grid})
    if density:
        sizes = [(x[1] - x[0]) / (n_bins - 2) for x in limits]
        box_size = np.prod(sizes)
        agg = agg / box_size  # type: ignore # this is a valid op
    if weight == "outer":
        weights = pd.Series(1, index=agg.index)
    elif weight == "inner":
        weights = pd.Series(1, index=agg.dropna().index)
    elif weight == "left":
        weights = left_grid
    elif weight == "right":
        weights = right_grid
    elif weight == "mix":
        weights = (agg["left"].fillna(baseline) + agg["right"].fillna(baseline)) / 2
    elif weight == "uright":
        weights = pd.Series(1, index=right_grid.index)
    elif weight == "uleft":
        weights = pd.Series(1, index=left_grid.index)
    else:
        raise ValueError()
    agg = agg.fillna(baseline)
    weights = weights / weights.sum()
    agg["weights"] = weights
    agg = agg.dropna()
    agg["error"] = loss(agg["left"].to_numpy(), agg["right"].to_numpy())
    agg["weighted_error"] = agg["error"] * agg["weights"]
    if return_table:
        return agg
    else:
        return agg["weighted_error"].sum()


def helper_metrics_2d(target_x, target_y, sampled_x, sampled_y, limits_x, limits_y, n_bins=64):
    data_df = pd.DataFrame({"x": target_x.flatten(), "y": target_y.flatten()})
    model_df = pd.DataFrame({"x": sampled_x.flatten(), "y": sampled_y.flatten()})

    rms_fe_sq_error = grid_pointwise(
        left=model_df,
        right=data_df,
        n_bins=n_bins,
        density=False,
        baseline=1e-6,
        limits=[limits_x, limits_y],
        loss=fe_sq_error,
        columns=["x", "y"],
        weight="mix",
        outlier_check=False,
    )

    rms_mjs_error = grid_pointwise(
        left=model_df,
        right=data_df,
        n_bins=n_bins,
        density=False,
        baseline=1e-6,
        limits=[limits_x, limits_y],
        loss=mjs_error,
        columns=["x", "y"],
        weight="mix",
        outlier_check=False,
    )

    return rms_fe_sq_error, rms_mjs_error


def phi_psi_metrics(target_phi, target_psi, sampled_phi, sampled_psi, n_bins=64):
    return helper_metrics_2d(
        target_phi,
        target_psi,
        sampled_phi,
        sampled_psi,
        limits_x=[-jnp.pi, jnp.pi],
        limits_y=[-jnp.pi, jnp.pi],
        n_bins=n_bins,
    )
