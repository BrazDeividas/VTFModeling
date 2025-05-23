from data_utils import zscore, smooth_data
import numpy as np
import torch

def prepare_predictions_dataset(sensor_data_matrix, train_split = 0.8, sequence_len = 24, prediction_len = 6):

    normalized_sensor_data = zscore(sensor_data_matrix, sensor_data_matrix.mean(), sensor_data_matrix.std())
    smoothed_sensor_data = smooth_data(normalized_sensor_data)

    train_data = smoothed_sensor_data[:int(sensor_data_matrix.shape[0]*train_split)].T
    test_data = smoothed_sensor_data[int(sensor_data_matrix.shape[0]*train_split):].T

    X_train, Y_train, X_test, Y_test = build_features_labels(sequence_len, prediction_len, train_data, test_data)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], Y_test.shape[2])
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1],Y_train.shape[2])

    return X_train, Y_train, X_test, Y_test


def build_features_labels(sequence_len, prediction_len, train_data, test_data):
    X_train, Y_train, X_test, Y_test = [], [], [], []

    for i in range(train_data.shape[1] - int(sequence_len + prediction_len - 1)):
        a = train_data[:, i : i + sequence_len + prediction_len]
        X_train.append(a[:, :sequence_len])
        Y_train.append(a[:, sequence_len:sequence_len+prediction_len])

    for i in range(test_data.shape[1] - int(sequence_len + prediction_len - 1)):
        b = test_data[:, i : i + sequence_len + prediction_len]
        X_test.append(b[:, :sequence_len])
        Y_test.append(b[:, sequence_len:sequence_len+prediction_len])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

def prepare_edge_index_and_attributes(adjacency_matrix, device):
    edges = np.nonzero(adjacency_matrix)
    edges = np.vstack([edges, adjacency_matrix[edges]])
    edge_index = edges[:2, :].astype(float)
    edge_attr = edges[2, :].astype(float)

    # Reshape edge attributes to [num_edges, 1] for the model 
    edge_attr = edge_attr.reshape(-1, 1)
    
    # Convert to tensors and move to device
    edge_index = torch.from_numpy(edge_index).type(torch.long).to(device)
    edge_attr = torch.from_numpy(edge_attr).type(torch.float).to(device)

    return edge_index, edge_attr

def prepare_edge_weights(adjacency_matrix, device):
    """Extract edge weights from adjacency matrix"""
    edges = np.nonzero(adjacency_matrix)
    edge_weights = adjacency_matrix[edges]
    edge_weights = torch.from_numpy(edge_weights).type(torch.float).to(device)
    return edge_weights

def prepare_prediction_dataloader(X_train, Y_train, X_test, Y_test, device, batch_size = 32):

    train_input = np.array(X_train)
    train_target = np.array(Y_train)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(device)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, drop_last=True)

    test_input = np.array(X_test)
    test_target = np.array(Y_test)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader