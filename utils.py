import torch
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.auto import tqdm
import random
from torch.utils.data import SubsetRandomSampler
from data.data_utils import ProtectedDataset
from distances.binary_distances import BinaryDistance
from distances.distance import Distance

def load_model(model, dataset_name, trainer_name, use_sensitive_attr, sensitive_vars, id, note='', path='no_norm'):
    root_dir = os.path.join('trained_models', path)
    file_name = f'MLP_{dataset_name}_{trainer_name}_{"all-features" if use_sensitive_attr else "without-"+"-".join(sensitive_vars)}_{id}{note}'
    model.load(os.path.join(root_dir, file_name))

def get_data(data, rand_seed, sensitive_vars):
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)

    X, y, sensitive_idxs = data.load_data(sensitive_vars=sensitive_vars)

    # randomly split into train/test splits
    total_samples = len(X)
    train_size = int(total_samples * 0.8)

    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataset = ProtectedDataset(X, y, sensitive_idxs)
    train_loader = DataLoader(dataset, batch_size=512, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1000, sampler=test_sampler)

    return dataset, train_loader, test_loader

class UnfairMetric():
    def __init__(self, dx: Distance, dy: BinaryDistance, epsilon: float) -> None:
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
    
    def is_unfair(self, x1, x2, y1, y2):
        return (self.dy(y1, y2).item() > self.dx(x1, x2).item()*self.epsilon)
    
def get_L_matrix(all_X, all_pred, dx, dy):
    ds = TensorDataset(all_X, all_pred)
    dl = DataLoader(ds, batch_size=3000, shuffle=False)
    L = []
    for b in tqdm(dl):
        dxs = dx(b[0], all_X, itemwise_dist=False)
        dys = dy(b[1], all_pred, itemwise_dist=False)

        L_batch = (dys/dxs).squeeze()
        L.append(L_batch)
    L = torch.concat(L, dim=0)
    return L

def decide_label_by_majority_voting(pair, model, data_gen, num_data_around):
    def mv(datapoint):
        data_around = data_gen.generate_around_datapoint(datapoint, num_data_around)
        pred = model.get_prediction(data_around)
        assert torch.all(torch.logical_or(pred ==0, pred == 1))
        n_pred = pred.numel()
        n_pred_pos = pred.sum()
        return n_pred_pos / n_pred - 0.5
    p1, p2 = mv(pair[0]), mv(pair[1])
    p = p1 if abs(p1) > abs(p2) else p2
    # print('proportion of positive prediction', p1 + 0.5, p2 + 0.5)
    return int(p > 0)

def add_data_to_dataset(old_train_loader, new_X, new_label, device):
    new_label = new_label.int()

    all_X_train, all_y_train = [], []
    for x, y in old_train_loader:
        all_X_train.append(x)
        all_y_train.append(y)
    all_X_train = torch.concat(all_X_train, dim=0).to('cpu')
    all_y_train = torch.concat(all_y_train, dim=0).to('cpu')

    all_X_train = torch.concat([all_X_train, new_X], dim=0).detach().to(device)
    all_y_train = torch.concat([all_y_train, new_label], dim=0).detach().to(device)

    retrain_dataset = TensorDataset(all_X_train, all_y_train)
    retrain_dataloader = DataLoader(retrain_dataset, batch_size=512)
    return retrain_dataloader