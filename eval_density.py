import torch
import json
import sys
import numpy as np
from metrics import test_acc
from read_dataset import ImageDataset

# SHELL ARGUMENT PARSING
MEAN = sys.argv[1]

# LOAD MODEL
model = torch.load(f'dataset/density_experiment/model_{MEAN}.pt')
model.eval()

# LOAD DATASETS
means = np.arange(10, 100, 10)
test_loaders = []
for mean in means:
    dataset = ImageDataset(data_folder=f'dataset/density_experiment/dataset_{mean}',
                           label_file=f'dataset/density_experiment/dataset_{mean}/image_data.json')
    testset = torch.utils.data.Subset(dataset, range(8000, 10000))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    test_loaders.append(test_loader)

# EVALUATE MODEL
accs = []
for loader in test_loaders:
    acc = test_acc(model, loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    accs.append(acc)
print(f'ACCS: {accs}')

# SAVE ACCURACIES
with open('dataset/density_experiment/evals.json', 'r') as f:
    acc_dict = json.load(f)
acc_dict[MEAN] = accs
with open('dataset/density_experiment/evals.json', 'w') as f:
    json.dump(acc_dict, f)
