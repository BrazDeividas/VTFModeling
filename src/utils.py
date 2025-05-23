import pickle as pkl
import torch

def load_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")