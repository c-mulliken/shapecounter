import torch
import json
import sys
import numpy as np
from metrics import test_acc
from read_dataset import ImageDataset

# SHELL ARGUMENT PARSING
MODEL_MEAN = sys.argv[1]

# LOAD MODEL
model = torch.load(f'dataset/density_experiment/model_{MODEL_MEAN}.pt')
model.eval()

# LOAD DATASETS
sizes = [10, 50, 100]
margins = [-8, -4, -2, 0, 2, 4, 8]
test_loaders = []
for size in sizes:
    size_loaders = []
    for margin in margins:
        testset = ImageDataset(data_folder=f'dataset/discrim_tests/size_{size}/margin_{margin}',
                               label_file=f'dataset/discrim_tests/size_{size}/margin_{margin}/image_data.json')
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        size_loaders.append(test_loader)
    test_loaders.append(size_loaders)

# EVALUATE MODEL
accs = {}
for i, size_loader in enumerate(test_loaders):
    size_accs = []
    for loader in size_loader:
        acc = test_acc(model, loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        size_accs.append(acc)
    accs[sizes[i]] = size_accs
    print(f'MODEL {MODEL_MEAN} | SIZE {sizes[i]} | ACCS: {size_accs}')
print(f'MODEL {MODEL_MEAN} | FULL ACCS: {accs}')

# SAVE ACCURACIES
with open('dataset/discrim_tests/evals.json', 'r') as f:
    acc_dict = json.load(f)
acc_dict[MODEL_MEAN] = accs
with open('dataset/discrim_tests/evals.json', 'w') as f:
    json.dump(acc_dict, f)