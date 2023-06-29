import time
from collections import OrderedDict

import torch
import torch.utils.data

import os, sys
import flwr as fl

import pandas as pd
import numpy as np

import helper as hl
import dataset as ds
import models as ml
import train as tr
import argparse


def load_data(params, bins=np.arange(0, 36, 5) * 60):
    print(params)
    data_path = f'./data/pkl/{params["data"]}_dataset_'+\
                f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{params["crs"]}_'+\
                f'{"dspeed" if params["dspeed"] else ""}_'+\
                f'{"dcourse" if params["dcourse"] else ""}.traj_delta_windows.pickle'
    
    # ## Parse Trajectories
    traj_delta_windows = pd.read_pickle(data_path)
    print(f'Loaded Trajectories from {os.path.basename(data_path)}')

    # ## Split Trajectories to Train/Dev/Test Sets
    #   * #### Split trajectories train/dev/test sets
    look_discrete = pd.cut(traj_delta_windows.samples.apply(lambda l: l[-1, -1]), bins=bins).rename('labels').to_frame().reset_index(drop=True)
    train_indices, dev_indices, test_indices = ds.timeseries_train_test_split(look_discrete, dev_size=0.25, test_size=0.25, stratify=params["strat"])

    #   * #### Create unified train/dev/test dataset(s)
    train_delta_windows = traj_delta_windows.iloc[train_indices].copy()
    dev_delta_windows = traj_delta_windows.iloc[dev_indices].copy()
    test_delta_windows = traj_delta_windows.iloc[test_indices].copy()

    train_dataset = ds.VRFDataset(train_delta_windows)
    dev_dataset, test_dataset = ds.VRFDataset(dev_delta_windows, scaler=train_dataset.scaler),\
                                ds.VRFDataset(test_delta_windows, scaler=train_dataset.scaler)

    train_loader, dev_loader, test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["bs"], shuffle=True, collate_fn=train_dataset.pad_collate),\
                                            torch.utils.data.DataLoader(dev_dataset,   batch_size=params["bs"], shuffle=False, collate_fn=dev_dataset.pad_collate),\
                                            torch.utils.data.DataLoader(test_dataset,  batch_size=params["bs"], shuffle=False, collate_fn=test_dataset.pad_collate)
   
    return data_path, train_loader, dev_loader, test_loader


class VRFClient(fl.client.NumPyClient):
    def __init__(self, save_dir, device, train_loader, dev_loader, test_loader, model_params, use_atten=False, load_check=False):
        self.model = ml.VesselRouteForecasting(**model_params) if not use_atten else ml.AttentionVRF(**model_params)
        self.device = device
        self.criterion = tr.RMSELoss(eps=1e-4)

        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.train_loader, self.dev_loader, self.test_loader = train_loader, dev_loader, test_loader
        self.train_losses = []
        self.dev_losses = []

        self.num_examples = dict(
            training_set=len(self.train_loader.dataset),
            test_set=len(self.test_loader.dataset)
        )
        self.save_path = save_dir

        if load_check:
            print('Loading Latest Checkpoint...')
            model_params = torch.load(self.save_path)
            self.model.load_state_dict(model_params['model_state_dict'])
            self.optimizer.load_state_dict(model_params['optimizer_state_dict'])
            self.model.mu, self.model.sigma = torch.Tensor(model_params['scaler'].mean_[:2]),\
                                              torch.Tensor(model_params['scaler'].scale_[:2])
            self.train_losses = model_params['loss']
            self.dev_losses = model_params['dev_loss']

    def get_parameters(self, **kwargs):
        print('@VRFClient.get_parameters()', f'{kwargs=}', sep='\t|\t')
        return hl.get_parameters(self.model)
        # return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        return hl.set_parameters(self.model, parameters)
        # params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print('@VRFClient.get_parameters()', f'{config=}', sep='\t|\t')
        self.set_parameters(parameters)
        train_loss, dev_loss = tr.train_model(
            self.model, self.device, self.criterion, self.optimizer,
            n_epochs=1,
            train_loader=self.train_loader,
            dev_loader=self.dev_loader,
            early_stop=False,
        )
        fit_tldr = dict(
            train_loss=float(train_loss[-1]),
            dev_loss=float(dev_loss[-1]),
        )
        self.train_losses.append(fit_tldr['train_loss'])
        self.dev_losses.append(fit_tldr['dev_loss'])

        # Save Current Client
        kwargs = dict({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses,
            'dev_loss': self.dev_losses,
            'scaler': train_loader.dataset.scaler,
        })
        tr.save_model(self.model, self.save_path, **kwargs)
        
        return self.get_parameters(), self.num_examples['training_set'], fit_tldr

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, test_acc = tr.evaluate_model(self.model, self.device, self.criterion, self.test_loader)

        eval_tldr = dict(
            test_loss=float(test_loss),
            test_acc=float(np.mean(test_acc)),
        )
        return np.float64(test_loss), self.num_examples['test_set'], eval_tldr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Distributed VRF Worker')
    parser.add_argument('--data', help='Select Dataset', choices=['brest', 'norway', 'piraeus', 'mt'], type=str, required=True)
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--strat', help='Create Stratifies Train/Dev/Test Datasets', action='store_true')
    parser.add_argument('--atten', help='Use Attention Mechanism', action='store_true')
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--bs', help='Batch Size', default=1, type=int, required=False)
    parser.add_argument('--load_check', help='Continue from Latest Epoch', action="store_true")
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)
    parser.add_argument('--perfl', help='Use Weighted (i.e., Personalized) Federated Learning', action='store_true')

    args = parser.parse_args()
    params = dict(
        **vars(args),
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
    print(f'{params["input_feats"]=}')

    data_path, train_loader, dev_loader, test_loader = load_data(params)
    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')

    model_params = dict(
        input_size=len(params['input_feats']),
        scale=dict(
            sigma=torch.Tensor(train_loader.dataset.scaler.scale_[:2]), 
            mu=torch.Tensor(train_loader.dataset.scaler.mean_[:2])
        ),
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        fc_layers=[150,]
    )

    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                 f'{"atten-" if args.atten else ""}lstm_{model_params["num_layers"]}_'+\
                 f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}'+\
                 f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{args.crs}_'+\
                 f'{"dspeed" if args.dspeed else ""}_'+\
                 f'{"dcourse" if args.dcourse else ""}_'+\
                 f'batchsize_{args.bs}__'+\
                 f'{args.data}_dataset_{"stratified" if args.strat else ""}.flwr_local.pth'
    save_path = os.path.join('.', 'data', 'pth', f'{"perfl" if args.perfl else "fl"}_experiments', model_name)
    print(save_path)

    client = VRFClient(
        save_path, device, 
        train_loader, dev_loader, test_loader,
        model_params, 
        args.atten, args.load_check
    )
    print(f"[::]:{args.port}")
    fl.client.start_numpy_client(server_address=f"[::]:{args.port}", client=client)
