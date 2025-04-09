import numpy as np
import torch
import argparse

from pdgrapher import Dataset, PDGrapher, Trainer

import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """

    parser = argparse.ArgumentParser(description="Train a model using the PDGrapher framework.")
    parser.add_argument('--forward_path', type=str, required=True, help="Path to the forward data.")
    parser.add_argument('--backward_path', type=str, required=True, help="Path to the backward data.")
    parser.add_argument('--edge_index_path', type=str, required=True, help="Path to the edge_index data.")
    parser.add_argument('--splits_path', type=str, required=True, help="Path to the splits data.")
    parser.add_argument('--use_forward_data', action='store_true', help="Whether to use forward data")
    parser.set_defaults(use_forward_data=False)
    
    args = parser.parse_args()

    dataset = Dataset(
        forward_path=args.forward_path,
        backward_path=args.backward_path,
        splits_path=args.splits_path
    )

    edge_index = torch.load(args.edge_index_path)
    
    # model = PDGrapher(edge_index, model_kwargs={
    #     "n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars()
    #     })
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
        "dim_gnn": 8, "num_vars": dataset.get_num_vars()
        })

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=args.use_forward_data, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=True, log_test=True
    )

    # Iterate over all of the folds and train on each one
    # model_performances = trainer.train_kfold(model, dataset, n_epochs = 1)
    model_performances = trainer.train_kfold(model, dataset, n_epochs = 5)

    print(model_performances)
    with open(f"/home/b-evelyntong/hl/lincs_lvl3_oe/multifold_final.txt", "w") as f:
        f.write(str(model_performances))


if __name__ == "__main__":
    main()