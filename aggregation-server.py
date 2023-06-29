"""Flower server example."""
import os
import argparse
from copy import deepcopy

import torch
import flwr as fl

import models as ml
import train as tr
import strategy as st


if __name__ == "__main__":
    def get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    parser = argparse.ArgumentParser(prog='VRF Aggregation Server')
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--atten', help='Use Attention Mechanism', action='store_true')
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--num_rounds', help='#FL Rounds (default: 170)', default=170, type=int, required=False)
    parser.add_argument('--load_check', help='Continue from Latest Epoch', action="store_true")
    parser.add_argument('--perfl', help='Use Weighted (i.e., Personalized) Federated Learning', action='store_true')
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)
    args = parser.parse_args()
    
    params = dict(
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

    model_params = dict(
        input_size=len(params['input_feats']),
        scale=None,
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        fc_layers=[150,]
    )

    model = ml.VesselRouteForecasting(**model_params) if not args.atten else ml.AttentionVRF(**model_params)
    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                 f'{"atten-" if args.atten else ""}lstm_{model_params["num_layers"]}_'+\
                 f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_'+\
                 f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{args.crs}_'+\
                 f'{"dspeed" if args.dspeed else ""}_'+\
                 f'{"dcourse" if args.dcourse else ""}_'+\
                 f'.flwr_global_epoch{{0}}.pth'
    
    save_path = os.path.join('.', 'data', 'pth', f'{"perfl" if args.perfl else "fl"}_experiments', model_name)
    print(save_path)

    if args.load_check:
        print('Loading Latest Checkpoint...')
        model_params = torch.load(save_path.format(args.num_rounds))
        model.load_state_dict(model_params['model_state_dict'])

    if not args.perfl:
        print('Using ```FedAvg``` Aggregation Strategy')
        strategy = st.FedAdamVRF(
            model=deepcopy(model),
            save_dir=save_path,
            num_rounds=args.num_rounds,
            load_check=args.load_check,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
            eta=1e-3,
            eta_l=1e-3,
            beta_1=0.9,
            beta_2=0.999,
        )
    else:
        print('Using ```qFedAvg``` Aggregation Strategy')
        strategy = st.qFedAvgVRF(
            model=deepcopy(model),
            save_dir=save_path,
            num_rounds=args.num_rounds,
            load_check=args.load_check,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            evaluate_fn=st.get_evaluate_fn(model, tr.RMSELoss(eps=1e-4))
        )

    fl.server.start_server(
        server_address=f"[::]:{args.port}", 
        config=fl.server.ServerConfig(num_rounds=args.num_rounds), 
        strategy=strategy
    )
