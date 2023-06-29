import flwr.server.strategy
import torch
import flwr as fl
import pandas as pd
import numpy as np

from flwr.common import FitRes, Parameters, Scalar, EvaluateRes
from flwr.server.strategy import FedAvg, FedAdam, QFedAvg
from flwr.server.client_proxy import ClientProxy

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import helper as hl
import dataset as ds
import train as tr


def model_checkpoint(obj, rnd, results, aggregated_weights):
    hl.set_parameters(obj.model, fl.common.parameters_to_ndarrays(aggregated_weights[0]))

    obj.train_loss_aggregated.append(hl.weighted_sum(results, 'train_loss'))
    obj.dev_loss_aggregated.append(hl.weighted_sum(results, 'dev_loss'))

    kwargs = dict({
        'epoch': rnd,
        'loss': obj.train_loss_aggregated,
        'dev_loss': obj.dev_loss_aggregated
    })
    # tr.save_model(self.model, self.save_path, **kwargs)
    tr.save_model(obj.model, obj.save_path.format(rnd), **kwargs)

    return aggregated_weights


def get_evaluate_fn(model, criterion):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    data_path = f'./data/pkl/mt_dataset_window_1024_stride_1024_crs_3857__.traj_delta_windows.pickle'
    delta_series = pd.read_pickle(data_path)

    # ## Split Trajectories to Train/Dev/Test Sets
    look_discrete = pd.cut(
        delta_series.samples.apply(lambda l: l[-1, -1]), 
        bins=np.arange(0, 36, 5) * 60
    ).rename('labels').to_frame().reset_index(drop=True)
    train_indices, _, test_indices = ds.timeseries_train_test_split(look_discrete, dev_size=0.25, test_size=0.25, stratify=True)

    #   * #### Create unified train/dev/test dataset(s)
    train_dataset = ds.VRFDataset(delta_series.iloc[train_indices])
    test_dataset = ds.VRFDataset(delta_series.iloc[test_indices], scaler=train_dataset.scaler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.pad_collate)

    model.mu, model.sigma = torch.Tensor(train_dataset.scaler.mean_[:2]),\
                            torch.Tensor(train_dataset.scaler.scale_[:2])

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters_ndarrays, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print(
            '@get_evaluate_fn.<locals>.evaluate()', 
            f'{server_round=}, {config=}', 
            f'{model.mu=}', f'{model.sigma=}', 
            sep='\t|\t'
        )
        hl.set_parameters(model, parameters_ndarrays)
        eval_loss, eval_acc = tr.evaluate_model(model, torch.device('cpu'), criterion, test_loader)
        
        return float(eval_loss), {"accuracy": float(np.mean(eval_acc))}

    return evaluate


class FedAdamVRF(FedAdam):
    def __init__(self, model, save_dir, num_rounds, load_check, ndigits=10, **kwargs):
        self.model = model
        self.ndigits = ndigits
        self.save_path = save_dir
        self.train_loss_aggregated = []
        self.dev_loss_aggregated = []

        if load_check:
            model_params = torch.load(self.save_path.format(num_rounds))
            self.train_loss_aggregated = model_params['loss']
            self.dev_loss_aggregated = model_params['dev_loss']

        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            model_checkpoint(self, server_round, results, aggregated_weights)

        return aggregated_weights

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracy_aggregated = hl.weighted_sum(results, 'test_acc')
        # print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        loss_aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)

        return round(loss_aggregated, ndigits=self.ndigits), \
               {**metrics, 'accuracy': round(accuracy_aggregated, ndigits=self.ndigits)}


class qFedAvgVRF(QFedAvg):
    def __init__(self, model, save_dir, num_rounds, load_check, ndigits=10, **kwargs):
        self.model = model
        self.ndigits = ndigits
        self.save_path = save_dir
        self.train_loss_aggregated = []
        self.dev_loss_aggregated = []

        if load_check:
            model_params = torch.load(self.save_path.format(num_rounds))
            self.train_loss_aggregated = model_params['loss']
            self.dev_loss_aggregated = model_params['dev_loss']

        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            model_checkpoint(self, server_round, results, aggregated_weights)

        return aggregated_weights

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracy_aggregated = hl.weighted_sum(results, 'test_acc')
        # print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        loss_aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)

        return round(loss_aggregated, ndigits=self.ndigits), \
               {**metrics, 'accuracy': round(accuracy_aggregated, ndigits=self.ndigits)}