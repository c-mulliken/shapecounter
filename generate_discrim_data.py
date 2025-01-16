from generate_dataset import create_discrim_dataset

# HYPERPARAMETERS
BATCH_SIZE = 64
PATH = '/home/coby/Repositories/shapecounter/dataset/discrim_tests'

# CREATE DATASETS
sizes = [10, 50, 100]
margins = [-8, -4, -2, 0, 2, 4, 8]

for size in sizes:
    for margin in margins:
        create_discrim_dataset(2000, size, margin, f'{PATH}/size_{size}/margin_{margin}')