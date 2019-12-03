import os, torch, random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_np(t):
    # return t.cpu().detach().numpy()
    return t.detach().numpy()

def soft_voting(probs):
    _arrs = [probs[key] for key in probs]
    return np.mean(np.mean(_arrs, axis=1), axis=0)