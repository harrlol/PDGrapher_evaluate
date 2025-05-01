import os
import numpy as np
import torch
import argparse

from pdgrapher import Dataset, PDGrapher, Trainer

torch.set_num_threads(20)
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
    parser.add_argument('--embedding_path', type=str, required=False, help="Path to the precomputed embeddings.")
    parser.add_argument('--use_forward_data', action='store_true', help="Whether to use forward data")
    parser.add_argument('--n_epoch', type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument('--output_dir', type=str, default=".", help="Directory to save the model checkpoints (default: current directory)")
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
    # check if embedding_path is provided
    if args.embedding_path is not None:
        model = PDGrapher(edge_index, model_kwargs={
            "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
            "dim_gnn": 8, "num_vars": dataset.get_num_vars()
            }, precomputed_embeddings_path=args.embedding_path)
    else:
        model = PDGrapher(edge_index, model_kwargs={
            "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
            "dim_gnn": 8, "num_vars": dataset.get_num_vars()
            })

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=args.use_forward_data, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=True, log_test=True, logging_dir=args.output_dir
    )

    # Iterate over all of the folds and train on each one
    model_performances = trainer.train_kfold(model, dataset, n_epochs = args.n_epoch)

    print(model_performances)
    with open(os.path.join(args.output_dir, "multifold_final.txt"), "w") as f:
        f.write(str(model_performances))

if __name__ == "__main__":
    main()