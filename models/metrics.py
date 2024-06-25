import torch
from torch.utils.data import TensorDataset, DataLoader

def accuracy(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)
    corr, total = 0, 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)
        total += y.shape[0]
        corr += torch.sum(y_pred == y)

    score = corr / float(total)
    return score

def equally_opportunity(model, s_data_loader, device='cpu'):
    model.eval()
    model.to(device)

    n_real_pos_g0, n_real_pos_g1 = 0, 0
    n_true_pos_g0, n_true_pos_g1 = 0, 0

    for x, y, g in s_data_loader:
        pred = model.get_prediction(x)
        pred_pos_g0 = pred[torch.logical_and(y==1, g==0)]
        pred_pos_g1 = pred[torch.logical_and(y==1, g==1)]

        n_real_pos_g0 += pred_pos_g0.shape[0]
        n_real_pos_g1 += pred_pos_g1.shape[0]
        n_true_pos_g0 += pred_pos_g0.sum().item()
        n_true_pos_g1 += pred_pos_g1.sum().item()
    return abs(n_true_pos_g0/n_real_pos_g0 - n_true_pos_g1/n_true_pos_g1)

def individual_unfairness(model, unfair_metric, samples):
    batchsize=6000
    all_pred = model.get_prediction(samples)
    ds = TensorDataset(samples, all_pred)
    dl = DataLoader(ds, batch_size=batchsize, shuffle=False)
    L = []
    # if ds.__len__() > batchsize:
    #     dl = tqdm(dl)
    for b in dl:
        dxs = unfair_metric.dx(b[0], samples, itemwise_dist=False)
        dys = unfair_metric.dy(b[1], all_pred, itemwise_dist=False)
        L_batch = (dys/dxs).squeeze()
        L_vector = L_batch[torch.logical_not(torch.isnan(L_batch))].flatten()
        L.append(L_vector)
    L = torch.concat(L, dim=0)
    return L.mean().item()