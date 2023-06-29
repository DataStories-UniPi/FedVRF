#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

# In[3]:
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)

# In[4]:
import dataset as ds 
import models as ml
import train as tr
import helper as hl


# %%
#   * #### Create delta series
def create_delta_dataset(segment, time_name, speed_name, course_name, crs=3857, min_pts=22):
    if len(segment) < min_pts:
        return None

    segment.sort_values(time_name, inplace=True)
    
    delta_curr = segment.to_crs(crs)[segment.geometry.name].apply(lambda l: pd.Series(hl.shapely_coords_numpy(l), index=['dlon', 'dlat'])).diff()
    delta_curr_feats = segment[[speed_name, course_name]].diff().rename({speed_name:'dspeed_curr', course_name:'dcourse_curr'}, axis=1)
    delta_next = delta_curr.shift(-1)
    delta_tau  = pd.merge(
        segment[time_name].diff().rename('dt_curr'),
        segment[time_name].diff().shift(-1).rename('dt_next'),
        right_index=True, 
        left_index=True
    )
    
    return delta_curr.join(delta_curr_feats).join(delta_tau).join(delta_next, lsuffix='_curr', rsuffix='_next').dropna(subset=['dt_curr', 'dt_next'])


# %%
#   * #### Create constant-length windows for ML model training
def traj_windowing(
    segment, 
    length_max=1024, 
    length_min=20,
    stride=512, 
    input_feats=['dlon_curr', 'dlat_curr', 'dt_curr', 'dt_next'], 
    output_feats=['dlon_next', 'dlat_next'], 
):
    traj_inputs, traj_labels = [], []
    
    # input_feats_idx = [segment.columns.get_loc(input_feat) for input_feat in input_feats]
    output_feats_idx = [segment.columns.get_loc(output_feat) for output_feat in output_feats]
        
    for ptr_curr in range(0, len(segment), stride):
        segment_window = segment.iloc[ptr_curr:ptr_curr+length_max].copy()     

        if len(segment_window) < length_min:
            break

        traj_inputs.append(segment_window[input_feats].values)
        traj_labels.append(segment_window.iloc[-1, output_feats_idx].values)
    
    return pd.Series([traj_inputs, traj_labels], index=['samples', 'labels'])


# %%
def transform_to_delta_series(trajectories, group_attrs, crs=3857, **kwargs):
    trajectories = gpd.GeoDataFrame(trajectories, crs=crs, geometry=gpd.points_from_xy(trajectories['lon'], trajectories['lat']))

    # # Create VRF training dataset
    return hl.applyParallel(
        trajectories.groupby(group_attrs, group_keys=True), 
        lambda l: create_delta_dataset(l, **kwargs)
    )


def split_to_train_dev_test(traj_delta, group_attrs, windowing_params, bins=np.arange(0, 36, 5) * 60, stratify=True):
    traj_delta_windows = hl.applyParallel(
        traj_delta.reset_index().groupby(group_attrs),
        lambda l: traj_windowing(l, **windowing_params),
    ).reset_index(level=-1)\
        .pivot(columns=['level_2'])\
        .rename_axis([None, None], axis=1)\
        .sort_index(axis=1, ascending=False)

    traj_delta_windows.columns = traj_delta_windows.columns.droplevel(0)
    traj_delta_windows = traj_delta_windows.explode(['samples', 'labels'])

    #   * #### Split trajectories train/dev/test sets
    look_discrete = pd.cut(traj_delta_windows.samples.apply(lambda l: l[-1, -1]), bins=bins).rename('labels').to_frame().reset_index(drop=True)
    train_indices, dev_indices, test_indices = ds.timeseries_train_test_split(look_discrete, dev_size=0.25, test_size=0.25, stratify=stratify)

    ix_series = pd.Series({'train':train_indices, 'dev':dev_indices, 'test':test_indices})
    return traj_delta_windows.iloc[train_indices], traj_delta_windows.iloc[dev_indices], traj_delta_windows.iloc[test_indices], ix_series


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized ("Share-All") VRF Worker')
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--atten', help='Use Attention Mechanism', action='store_true')
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--bs', help='Batch Size', default=1, type=int, required=False)
    args = parser.parse_args()


    #   * #### Drop invalid MMSIs
    mmsi_mid = pd.read_pickle('mmsi_mid.pickle')


    # BREST
    trajectories_brest = pd.read_csv('./data/brest-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
    trajectories_brest_mmsis = trajectories_brest['sourcemmsi'].unique()
    valid_mmsis = [mmsi for mmsi in trajectories_brest_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
    trajectories_brest = trajectories_brest.loc[trajectories_brest['sourcemmsi'].isin(valid_mmsis)].copy()
    
    trajectories_brest_delta = transform_to_delta_series(
        trajectories_brest, 
        ['sourcemmsi', 'id'], 
        **dict(
            time_name='t', 
            speed_name='speedoverground', 
            course_name='courseoverground', 
            crs=3857, 
            min_pts=20
        )
    )

    # NORWAY
    trajectories_norway = pd.read_csv('./data/oslo-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
    trajectories_norway_mmsis = trajectories_norway['mmsi'].unique()
    valid_mmsis = [mmsi for mmsi in trajectories_norway_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
    trajectories_norway = trajectories_norway.loc[trajectories_norway['mmsi'].isin(valid_mmsis)].copy()

    trajectories_norway_delta = transform_to_delta_series(
        trajectories_norway, 
        ['mmsi', 'id'], 
        **dict(
            time_name='t', 
            speed_name='sog', 
            course_name='cog', 
            crs=3857, 
            min_pts=20
        )
    )

    # PIRAEUS
    trajectories_piraeus = pd.read_csv('./data/piraeus-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
    trajectories_piraeus_delta = transform_to_delta_series(
        trajectories_piraeus, 
        ['mmsi', 'id'], 
        **dict(
            time_name='t', 
            speed_name='speed', 
            course_name='course', 
            crs=3857, 
            min_pts=20
        )
    )

    # Account for Duplicate Vessels between Brest and Norway
    trajectories_norway_delta.index = trajectories_norway_delta.index.set_levels(trajectories_norway_delta.index.levels[1] + len(trajectories_brest_delta), level=1)

    # Create Windowed Sequence(s) for each dataset
    windowing_params, bins = dict(
        length_min=18, 
        length_max=1024, 
        stride=1024,
        input_feats=[
            'dlon_curr', 
            'dlat_curr', 
            *(['dspeed_curr',] if args.dspeed else []),
            *(['dcourse_curr',] if args.dcourse else []),
            'dt_curr', 
            'dt_next'
        ]
    ), np.arange(0, 36, 5) * 60
    print(f'{windowing_params["input_feats"]=}')

    trajectories_brest_delta_train, trajectories_brest_delta_dev, trajectories_brest_delta_test, trajectories_brest_ix_series = split_to_train_dev_test(
        trajectories_brest_delta.rename_axis(index={'sourcemmsi':'mmsi'}), 
        ['mmsi', 'id'],
        windowing_params, 
        bins=bins,
        stratify=True
    )
    
    trajectories_norway_delta_train, trajectories_norway_delta_dev, trajectories_norway_delta_test, trajectories_norway_ix_series = split_to_train_dev_test(
        trajectories_norway_delta, 
        ['mmsi', 'id'],
        windowing_params, 
        bins=bins,
        stratify=False
    )
    
    trajectories_piraeus_delta_train, trajectories_piraeus_delta_dev, trajectories_piraeus_delta_test, trajectories_piraeus_ix_series = split_to_train_dev_test(
        trajectories_piraeus_delta, 
        ['mmsi', 'id'],
        windowing_params, 
        bins=bins,
        stratify=True
    )
    
    # Merge Datasets
    #   * #### Create unified train/dev/test dataset(s)
    train_delta_windows = pd.concat((
        trajectories_brest_delta_train,
        trajectories_norway_delta_train,
        trajectories_piraeus_delta_train
    ))

    dev_delta_windows = pd.concat((
        trajectories_brest_delta_dev,
        trajectories_norway_delta_dev,
        trajectories_piraeus_delta_dev
    ))

    test_delta_windows = pd.concat((
        trajectories_brest_delta_test,
        trajectories_norway_delta_test,
        trajectories_piraeus_delta_test
    ))

    #   * #### Save indices (for future reference)
    trajectories_brest_ix_series.to_pickle('./data/pkl/brest_dataset_train_dev_test_indices.pkl')
    trajectories_norway_ix_series.to_pickle('./data/pkl/norway_dataset_train_dev_test_indices.pkl')
    trajectories_piraeus_ix_series.to_pickle('./data/pkl/piraeus_dataset_train_dev_test_indices.pkl')


    #   * #### Visualize Train/Dev/Test Distribution
    fig, ax = plt.subplots(1,3, figsize=(20, 7))

    look_discrete_train, look_discrete_dev, look_discrete_test = pd.cut(
        train_delta_windows.samples.apply(lambda l: l[-1, -1]
    ), bins=bins).rename('labels').to_frame().reset_index(drop=True), pd.cut(
        dev_delta_windows.samples.apply(lambda l: l[-1, -1]
    ), bins=bins).rename('labels').to_frame().reset_index(drop=True), pd.cut(
        test_delta_windows.samples.apply(lambda l: l[-1, -1]
    ), bins=bins).rename('labels').to_frame().reset_index(drop=True)

    look_discrete_train.value_counts(sort=False).plot.bar(ax=ax[0], color='tab:blue')
    look_discrete_dev.value_counts(sort=False).plot.bar(ax=ax[1], color='tab:orange')
    look_discrete_test.value_counts(sort=False).plot.bar(ax=ax[2], color='tab:green')

    [ax_i.set_yscale('log') for ax_i in ax];
    [ax_i.bar_label(ax_i.containers[0]) for ax_i in ax];

    plt.savefig(f'./data/fig/delta_series_distribution_share-all.pdf', dpi=300, bbox_inches='tight')


    # # Create kinematic features' temporal sequence (i.e. training dataset)
    train_dataset = ds.VRFDataset(train_delta_windows)
    dev_dataset, test_dataset = ds.VRFDataset(dev_delta_windows, scaler=train_dataset.scaler),\
                                ds.VRFDataset(test_delta_windows, scaler=train_dataset.scaler)

    train_loader, dev_loader, test_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=train_dataset.pad_collate),\
                                            DataLoader(dev_dataset,   batch_size=args.bs, shuffle=False, collate_fn=dev_dataset.pad_collate),\
                                            DataLoader(test_dataset,  batch_size=args.bs, shuffle=False, collate_fn=test_dataset.pad_collate)


    # In[12]:
    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')

    model_params = dict(
        input_size=len(windowing_params['input_feats']),
        scale=dict(
            sigma=torch.Tensor(train_dataset.scaler.scale_[:2]), 
            mu=torch.Tensor(train_dataset.scaler.mean_[:2])
        ),
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        # fc_layers=[128,16,]
        fc_layers=[150,]
    )

    model = ml.VesselRouteForecasting(**model_params) if not args.atten else ml.AttentionVRF(**model_params)
    model.to(device)

    print(model)
    print(f'{device=}')

    criterion = tr.RMSELoss(eps=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                f'{"atten-" if args.atten else ""}lstm_{model_params["num_layers"]}_'+\
                f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
                f'{"dspeed" if args.dspeed else ""}_'+\
                f'{"dcourse" if args.dcourse else ""}_'+\
                f'batchsize_{args.bs}__share_all.pth'
    save_path = os.path.join('.', 'data', 'pth', model_name)
    print(model_name)

    tr.train_model(model, device, criterion, optimizer, 170, train_loader, dev_loader, early_stop=True, patience=19, path=save_path)


    # %%
    ## Load Best Model and calculate displacement Errors on the test set
    checkpoint = torch.load(save_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    tr.evaluate_model(model, device, criterion, test_loader, desc='[Best Model] Test Dataset...')
