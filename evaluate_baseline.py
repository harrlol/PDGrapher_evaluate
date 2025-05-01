import torch
from tqdm import tqdm
from pdgrapher import PDGrapher, Dataset
from pdgrapher._utils import get_thresholds

# Load baseline model
baseline_pert_model = torch.load("/home/b-evelyntong/hl/training_history/train_0425_baseline_2_30_A/oe/A375/_fold_1_perturbation_discovery.pt")
edge_index = torch.load("/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/oe/edge_index_A375.pt")
embedding_path = "/home/b-evelyntong/hl/embedding_matrix_oe.pt"
device = torch.device('cuda')

dataset = Dataset(
            forward_path="/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/oe/data_forward_A375.pt",
            backward_path="/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/oe/data_backward_A375.pt",
            splits_path="/home/b-evelyntong/hl/lincs_lvl3_oe/splits/oe/genetic/A375/random/2fold/splits.pt"
        )

thresholds = get_thresholds(dataset)
thresholds = {k: v.to(device) if v is not None else v for k, v in thresholds.items()}

dataset.prepare_fold(1)

(
    train_loader_forward, train_loader_backward,
    val_loader_forward, val_loader_backward,
    test_loader_forward, test_loader_backward
) = dataset.get_dataloaders(batch_size=1, num_workers=0)


model_baseline = PDGrapher(
    edge_index,
    model_kwargs={"n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8, "dim_gnn": 8, "num_vars": 7632})

model_baseline.perturbation_discovery.load_state_dict(baseline_pert_model['model_state_dict'], strict=True)
model_baseline.perturbation_discovery.edge_index = model_baseline.perturbation_discovery.edge_index.to(device)
model_baseline.perturbation_discovery = model_baseline.perturbation_discovery.to(device)
model_baseline.perturbation_discovery.eval()

# Save predictions
baseline_preds = []

for idx, data in enumerate(tqdm(test_loader_backward)):
    data = data.to(device)
    
    input_tensor = torch.cat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], dim=1)
    batch = data.batch
    mutations = data.mutations

    with torch.no_grad():
        out = model_baseline.perturbation_discovery(
            input_tensor, batch, mutilate_mutations=mutations, threshold_input=thresholds
        )

    baseline_preds.append(out.detach().cpu())

ground_truths = []
for data in test_loader_backward:
    data = data.to('cpu')
    ground_truths.append(data.intervention.squeeze(0).cpu())
torch.save(ground_truths, "/home/b-evelyntong/hl/ground_truths.pt")


torch.save(baseline_preds, "/home/b-evelyntong/hl/baseline_preds.pt")
print("Saved baseline predictions to baseline_preds.pt")
