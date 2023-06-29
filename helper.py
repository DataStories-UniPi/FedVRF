import rdp
import torch
import numpy as np
import pandas as pd

from shapely.geometry import Point
from collections import OrderedDict

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


# Get the coordinates of a Shapely Geometry (e.g. Point, Polygon, etc.) as NumPy array
shapely_coords_numpy = lambda l: np.array(*list(l.coords))


def temporal_segmentation(df, temporal_name, threshold=30 * 60, min_pts=10, output_name='traj_nr'):
    dt = df[temporal_name].diff()

    # Get splitting points
    L = dt.loc[dt >= threshold].index.values.tolist()
    idx = np.cumsum(np.in1d(df.index, L))

    # Drop splitting points
    out = pd.Series(idx, index=df.index, name=output_name, dtype=int)
    out.drop(L, inplace=True)

    # Drop trips with |points| < min_pts (Rejected Trips)
    out2 = out.groupby(out).filter(lambda l: len(l) >= min_pts)
    return out2


def _temporal_segmentation(df, temporal_name, threshold=30 * 60, min_pts=10, output_name='traj_nr'):
    traj_nrs = temporal_segmentation(df, temporal_name, threshold, min_pts, output_name)
    return df.join(traj_nrs, how='inner')


def create_delta_series(sdf):
    # Sort points by time (for safety)
    dataset = sdf.sort_values('timestamp').copy()

    # Convert coordinates to EPSG:3857
    dataset = dataset.to_crs(epsg=3857)
    dataset.drop(dataset.loc[dataset.timestamp.diff() < 1].index, axis=0, inplace=True)

    X = dataset[['timestamp', 'lon', 'lat']].diff()
    X.rename({'lon': 'dlon_curr_4326', 'lat': 'dlat_curr_4326', 'timestamp': 'dt_curr'}, axis=1, inplace=True)

    # Add deltas in meters
    X.loc[:, 'dlon_curr_3857'] = dataset.geometry.x.diff()
    X.loc[:, 'dlat_curr_3857'] = dataset.geometry.y.diff()

    # Add input pt2 - dt_next
    X.loc[:, 'dt_next'] = X.dt_curr.shift(-1)

    # Add next deltas (labels)
    X.loc[:, 'dlon_next_4326'] = X.dlon_curr_4326.shift(-1)
    X.loc[:, 'dlat_next_4326'] = X.dlat_curr_4326.shift(-1)

    X.loc[:, 'dlon_next_3857'] = X.dlon_curr_3857.shift(-1)
    X.loc[:, 'dlat_next_3857'] = X.dlat_curr_3857.shift(-1)

    # For diagnostics (debug features)
    debug_time = dataset[['timestamp', ]].shift().join(dataset[['timestamp', ]],
                                                       lsuffix='_prev', rsuffix='_curr')
    debug_time.rename({'timestamp_prev': 't_prev', 'timestamp_curr': 't_curr'}, axis=1, inplace=True)

    debug_4326 = dataset[['lon', 'lat']].shift().join(dataset[['lon', 'lat']],
                                                      lsuffix='_prev_4326', rsuffix='_curr_4326')

    pts_3857 = pd.DataFrame({'lon': dataset.geometry.x, 'lat': dataset.geometry.y})
    debug_3857 = pts_3857.shift().join(pts_3857, lsuffix='_prev_3857', rsuffix='_curr_3857')

    debug = debug_4326.join(debug_3857).join(debug_time)
    result = X.join(debug)[
        ['dlon_curr_3857', 'dlat_curr_3857', 'dt_curr', 'dt_next', 'dlon_next_3857', 'dlat_next_3857',
         'dlon_curr_4326', 'dlat_curr_4326', 'dlon_next_4326', 'dlat_next_4326',
         'lon_prev_4326', 'lat_prev_4326', 'lon_curr_4326', 'lat_curr_4326', 'lon_prev_3857', 'lat_prev_3857',
         'lon_curr_3857', 'lat_curr_3857',
         't_prev', 't_curr']]

    return pd.Series([result.dropna()], index=['trajectory_delta'])


def displacement_error(actual, pred):
    disp_err = []

    for rloc, rloc_pred in zip(actual, pred.squeeze(0)):
        disp_err.append(Point(*rloc).distance(Point(*rloc_pred)))

    return disp_err


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_sum(results, metric):
    # Weigh accuracy of each client by number of examples used
    metric_aggregated = [r.metrics[metric] * r.num_examples for _, r in results]
    examples = [r.num_examples for _, r in results]

    # Aggregate and print custom metric
    return sum(metric_aggregated) / sum(examples)


def applyParallel(df_grouped, fun, n_jobs=-1, **kwargs):
    '''
    Forked from: https://stackoverflow.com/a/27027632
    '''
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f'Scaling {fun} to {n_jobs} CPUs')

    df_grouped_names = df_grouped.grouper.names
    _fun = lambda name, group: (fun(group.drop(df_grouped_names, axis=1)), name)

    result, keys = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_fun)(name, group) for name, group in tqdm(df_grouped, **kwargs)
    ))
    return pd.concat(result, keys=keys, names=df_grouped_names)
