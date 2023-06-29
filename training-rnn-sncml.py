#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import sys

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

import config as cfg
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
# ### Instantiate Torch Dataset
class VRFDataset(Dataset):
    def __init__(self, data, scaler=None, max_dt=1920, dtype=np.float32):
        self.samples = data['samples'].values
        self.labels = data['labels'].values.ravel()
        self.lengths = [len(l) for l in self.samples]

        self.max_dt = max_dt
        self.dtype = dtype
        
        if scaler is None:
            self.scaler = StandardScaler().fit(np.concatenate(self.samples))
        else:
            self.scaler = scaler

    def pad_collate(self, batch):
        '''
        xx: Samples (Delta Trajectory), yy: Labels (Next Delta), ll: Lengths
        '''
        (xx, yy, ll) = zip(*batch)
        
        # Right Zero Padding with Zeroes (for delta trajectory)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, torch.stack(yy), torch.tensor(ll)
    
    def __getitem__(self, item):
        return torch.tensor(self.scaler.transform(self.samples[item]).astype(self.dtype)),\
               torch.tensor(self.labels[item].reshape(1, -1).astype(self.dtype)),\
               torch.tensor(self.lengths[item])

    def __len__(self):
        return len(self.labels)



# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized ("Share-All") VRF Worker')
    parser.add_argument('--data', help='Select Dataset', choices=['brest', 'norway', 'piraeus', 'mt'], type=str, required=True)
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--strat', help='Create Stratifies Train/Dev/Test Datasets', action='store_true')
    parser.add_argument('--atten', help='Use Attention Mechanism', action='store_true')
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--bs', help='Batch Size', default=1, type=int, required=False)
    args = parser.parse_args()


    #   * #### Drop invalid MMSIs
    mmsi_mid = pd.read_pickle('mmsi_mid.pickle')


    if args.data == 'brest':
        # BREST
        trajectories = pd.read_csv('./data/brest-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME = 'sourcemmsi', 'speedoverground', 'courseoverground', 't'

        trajectories_mmsis = trajectories[VESSEL_NAME].unique()
        valid_mmsis = [mmsi for mmsi in trajectories_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
        trajectories = trajectories.loc[trajectories.mmsi.isin(valid_mmsis)].copy()
    
    elif args.data == 'norway':
        # NORWAY
        trajectories = pd.read_csv('./data/oslo-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME = 'mmsi', 'sog', 'cog', 't'
    
        trajectories_mmsis = trajectories[VESSEL_NAME].unique()
        valid_mmsis = [mmsi for mmsi in trajectories_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
        trajectories = trajectories.loc[trajectories.mmsi.isin(valid_mmsis)].copy()

    elif args.data == 'piraeus':
        # PIRAEUS
        trajectories = pd.read_csv('./data/piraeus-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME = 'mmsi', 'speed', 'course', 't'

    elif args.data == 'mt':
        # MARINETRAFFIC
        trajectories = pd.read_csv('./data/mt-dataset/datasetpr_split_trajectories_sets_shuffle.csv')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME = 'mmsi', 'speed', 'course', 't'

        trajectories_mmsis = trajectories[VESSEL_NAME].unique()
        valid_mmsis = [mmsi for mmsi in trajectories_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
        trajectories = trajectories.loc[trajectories.mmsi.isin(valid_mmsis)].copy()


    trajectories = gpd.GeoDataFrame(trajectories, crs=3857, geometry=gpd.points_from_xy(trajectories['lon'], trajectories['lat']))

    params = dict(
        time_name=TIME_NAME, 
        speed_name=SPEED_NAME, 
        course_name=COURSE_NAME, 
        crs=3857, 
        min_pts=20
    )

    # # Create VRF training dataset
    traj_delta = hl.applyParallel(
        trajectories.groupby([VESSEL_NAME, 'id'], group_keys=True), 
        lambda l: create_delta_dataset(l, **params)
    )

    windowing_params = dict(
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
    )
    print(f'{windowing_params["input_feats"]=}')

    traj_delta_windows = hl.applyParallel(
        traj_delta.reset_index().groupby([VESSEL_NAME, 'id']),
        lambda l: traj_windowing(l, **windowing_params),
    ).reset_index(level=-1)\
        .pivot(columns=['level_2'])\
        .rename_axis([None, None], axis=1)\
        .sort_index(axis=1, ascending=False)

    traj_delta_windows.columns = traj_delta_windows.columns.droplevel(0)
    traj_delta_windows = traj_delta_windows.explode(['samples', 'labels'])

    # traj_delta_windows.to_pickle(
    #     f'./data/pkl/{args.data}_dataset_'+\
    #     f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
    #     f'{"dspeed" if args.dspeed else ""}_'+\
    #     f'{"dcourse" if args.dcourse else ""}.traj_delta_windows.pickle'
    # )

    #   * #### Split trajectories train/dev/test sets
    bins = np.arange(0, 36, 5) * 60
    # pdb.set_trace()
    look_discrete = pd.cut(traj_delta_windows.samples.apply(lambda l: l[-1, -1]), bins=bins).rename('labels').to_frame().reset_index(drop=True)
    train_indices, dev_indices, test_indices = ds.timeseries_train_test_split(look_discrete, dev_size=0.25, test_size=0.25, stratify=args.strat)

    #   * #### Visualize Train/Dev/Test Distribution
    fig, ax = plt.subplots(1,3, figsize=(20, 7))
    look_discrete.iloc[train_indices].value_counts(sort=False).plot.bar(ax=ax[0], color='tab:blue')
    look_discrete.iloc[dev_indices].value_counts(sort=False).plot.bar(ax=ax[1], color='tab:orange')
    look_discrete.iloc[test_indices].value_counts(sort=False).plot.bar(ax=ax[2], color='tab:green')

    [ax_i.set_yscale('log') for ax_i in ax];
    [ax_i.bar_label(ax_i.containers[0]) for ax_i in ax];

    pd.Series(
        {'train':train_indices, 'dev':dev_indices, 'test':test_indices}
    ).to_pickle(
        f'./data/pkl/{args.data}_dataset_train_dev_test_indices_{"stratified" if args.strat else ""}.pkl'
    )
    plt.savefig(f'./data/fig/delta_series_distribution_{args.data}_{"stratified" if args.strat else ""}.pdf', dpi=300, bbox_inches='tight')
    
    #   * #### Create unified train/dev/test dataset(s)
    train_delta_windows = traj_delta_windows.iloc[train_indices].copy()
    dev_delta_windows = traj_delta_windows.iloc[dev_indices].copy()
    test_delta_windows = traj_delta_windows.iloc[test_indices].copy()

    # # Create kinematic features' temporal sequence (i.e. training dataset)
    train_dataset = VRFDataset(train_delta_windows)
    dev_dataset, test_dataset = VRFDataset(dev_delta_windows, scaler=train_dataset.scaler),\
                                VRFDataset(test_delta_windows, scaler=train_dataset.scaler)

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

    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    criterion = tr.RMSELoss(eps=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                f'{"atten-" if args.atten else ""}lstm_{model_params["num_layers"]}_'+\
                f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
                f'{"dspeed" if args.dspeed else ""}_'+\
                f'{"dcourse" if args.dcourse else ""}_'+\
                f'batchsize_{args.bs}__'+\
                f'{args.data}_dataset_{"stratified" if args.strat else ""}.pth'
    save_path = os.path.join('.', 'data', 'pth', model_name)
    print(model_name)

    tr.train_model(model, device, criterion, optimizer, 170, train_loader, dev_loader, early_stop=True, patience=19, path=save_path)


    # %%
    ## Load Best Model and Get the f***ing Errors
    checkpoint = torch.load(save_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    tr.evaluate_model(model, device, criterion, test_loader, desc='[Best Model] Test Dataset...')
