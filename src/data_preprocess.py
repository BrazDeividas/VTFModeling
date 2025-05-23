import pandas as pd
import os
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from data_utils import find_nearest_node
import numpy as np
import torch
from math import radians, sin, cos, sqrt, atan2

def preprocess_sensor_data(sensor_data_path):
    sensor_data = pd.concat([pd.read_csv(os.path.join(sensor_data_path, f), delimiter = ';')
                            for f in os.listdir(sensor_data_path) 
                            if f.endswith('.csv')], ignore_index=True)
    
    #get Time nd Name columns
    sensor_cols = [col for col in sensor_data.columns if col not in ['Time', 'Name']] 
    
    #convert to numeric and fill na with 0
    sensor_data[sensor_cols] = sensor_data[sensor_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    #remove rows where all values are 0
    sensor_data_cleaned = sensor_data.loc[~(sensor_data == 0).all(axis=1)] 

    #convert Time to datetime
    sensor_data_slice = sensor_data_cleaned[['Name', 'Time', 'spd_proc']]
    sensor_data_slice['Time'] = pd.to_datetime(sensor_data_slice['Time'], format='%d.%m.%Y %H:%M:%S')

    #normalize Time to a timeseries
    sensor_data_slice['Time'] = sensor_data_slice['Time'] - sensor_data_slice['Time'].min()
    sensor_data_slice['Time'] = sensor_data_slice['Time'].dt.total_seconds() / 3600

    #only keep the sensor id from the Name column
    sensor_data_slice['Name'] = sensor_data_slice['Name'].str.extract(r'(\d{3,4})')

    return sensor_data_slice
    

def preprocess_sensor_position_data(sensor_positions_path):
    sensor_positions = pd.read_excel(sensor_positions_path)

    #only keep the sensor id from the Node column
    sensor_positions['Node'] = sensor_positions['Node'].str.extract('(\d+)')

    #convert coordinates to floats
    for coord in ['x', 'y']:
        sensor_positions[coord] = pd.to_numeric(sensor_positions[coord].str.replace(',', '.').astype(float))

    return sensor_positions

def remove_invalid_sensors(sensor_data, sensor_positions):
    valid_sensors = set(sensor_data['Name'].unique()) & set(sensor_positions['Node'])

    return valid_sensors

def group_sensor_data_by_intersection(sensor_data):
    #here name is the intersection id
    sensor_data = sensor_data.groupby(['Name', 'Time']).mean().reset_index()

    return sensor_data

def get_sensor_data_matrix(sensor_data, nan_value=0):
    sensor_data_matrix = sensor_data.pivot(index='Time', columns='Name', values='spd_proc').sort_index()
    sensor_data_matrix = np.nan_to_num(sensor_data_matrix, nan=nan_value)

    return sensor_data_matrix

def get_gpd_sensor_positions(sensor_positions):
    gdf_sensors = gpd.GeoDataFrame(
        sensor_positions,
        geometry=gpd.points_from_xy(sensor_positions.x, sensor_positions.y),
        crs='EPSG:4326')

    gdf_sensors = gdf_sensors.to_crs(epsg=3346)

    return gdf_sensors

def find_nearest_road_nodes_to_sensor_nodes(gdf_sensors, road_graph):
    gdf_sensors["nearest_node"] = gdf_sensors.geometry.apply(lambda x: find_nearest_node(road_graph, x))

    return gdf_sensors

## Road data loading

def load_road_data(road_data_path):
    road_data = gpd.read_file(road_data_path)

    #convert to epsg:3346
    if road_data.crs != "EPSG:3346":
        road_data = road_data.to_crs(epsg=3346)

    return road_data

def get_road_graph(road_data):
    G_roads = nx.Graph()

    for _, row in road_data.iterrows():
        line = row.geometry  # Get the LINESTRING geometry
        coords = list(line.coords)  # Extract coordinates as a list of (x, y) tuples
        for i in range(len(coords) - 1):
            start = coords[i]  # Start point of the segment
            end = coords[i + 1]  # End point of the segment
            length = LineString([start, end]).length  # Length of the segment
            G_roads.add_edge(start, end, 
                           weight=length,
                           danga=row.get('danga', 0),
                           oneway=row.get('oneway', 0),
                           kategor=row.get('kategor', 0))  # Add edge with weight and road attributes

    return G_roads

def compute_knn_distance_matrix(road_graph, node_mapping, k=10, max_distance=5.0):
    """
    Compute distance matrix using K-Nearest Neighbors approach
    Args:
        road_graph: NetworkX graph of the road network
        node_mapping: Dictionary mapping node coordinates to indices
        k: Number of nearest neighbors to consider
        max_distance: Maximum distance in kilometers to consider for connections
    Returns:
        distance_matrix: Sparse distance matrix [num_nodes, num_nodes]
    """
    num_nodes = len(node_mapping)
    # Initialize sparse distance matrix
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Convert node_mapping to reverse lookup for efficiency
    reverse_mapping = {node: idx for node, idx in node_mapping.items()}
    
    print("Computing KNN distances...")
    for i, (node1, idx1) in enumerate(node_mapping.items()):
        if i % 100 == 0:
            print(f"Processing node {i}/{num_nodes}")
            
        # Get all nodes within max_distance
        try:
            # Use NetworkX's single_source_dijkstra_path_length for efficient distance computation
            distances = nx.single_source_dijkstra_path_length(
                road_graph, 
                node1, 
                cutoff=max_distance*1000  # Convert km to meters
            )
            
            # Sort nodes by distance and take top k
            sorted_nodes = sorted(distances.items(), key=lambda x: x[1])[:k+1]  # +1 because node1 is included
            
            # Fill distance matrix
            for node2, distance in sorted_nodes:
                if node2 in reverse_mapping:
                    idx2 = reverse_mapping[node2]
                    # Convert distance to kilometers
                    distance_matrix[idx1, idx2] = distance / 1000
                    distance_matrix[idx2, idx1] = distance / 1000  # Symmetric
                    
        except nx.NetworkXNoPath:
            continue
    
    return distance_matrix

def create_sparse_adjacency_matrix(distance_matrix, threshold=0.1):
    """
    Create sparse adjacency matrix from distance matrix
    Args:
        distance_matrix: Distance matrix [num_nodes, num_nodes]
        threshold: Distance threshold for creating edges
    Returns:
        sparse_adj_matrix: Sparse adjacency matrix
    """
    # Create adjacency matrix based on distance threshold
    adj_matrix = (distance_matrix > 0) & (distance_matrix <= threshold)
    
    # Convert to sparse matrix format
    from scipy import sparse
    sparse_adj_matrix = sparse.csr_matrix(adj_matrix)
    
    return sparse_adj_matrix

def prepare_edge_index_and_attributes_sparse(sparse_adj_matrix, distance_matrix, device):
    """
    Prepare edge index and attributes from sparse adjacency matrix
    Args:
        sparse_adj_matrix: Sparse adjacency matrix
        distance_matrix: Distance matrix
        device: Device to run the model on
    Returns:
        edge_index: Edge indices tensor
        edge_attr: Edge attributes tensor
    """
    # Get edge indices from sparse matrix
    edges = sparse_adj_matrix.nonzero()
    edge_index = np.vstack([edges[0], edges[1]])
    
    # Get corresponding distances
    edge_attr = distance_matrix[edges[0], edges[1]]
    
    # Convert to tensors
    edge_index = torch.from_numpy(edge_index).type(torch.long).to(device)
    edge_attr = torch.from_numpy(edge_attr).type(torch.float).to(device)
    
    return edge_index, edge_attr

def get_all_road_nodes(road_graph):
    """
    Extract all nodes from the road network
    Returns:
        all_nodes: List of all node coordinates
        node_mapping: Dictionary mapping node coordinates to node indices
    """
    all_nodes = list(road_graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
    return all_nodes, node_mapping

def create_full_network_matrix(sensor_data_matrix, road_graph, gdf_sensors, node_mapping):
    """
    Create a matrix for the full network, initializing non-sensor nodes with zeros
    Args:
        sensor_data_matrix: Data from sensor locations [num_time_steps, num_sensors] as numpy array
        road_graph: NetworkX graph of the road network
        gdf_sensors: GeoDataFrame with sensor positions (Node as index)
        node_mapping: Dictionary mapping node coordinates to indices
    Returns:
        full_network_matrix: Matrix for all nodes [num_time_steps, num_nodes]
        sensor_indices: Indices of sensor locations in the full network
    """
    num_time_steps = sensor_data_matrix.shape[0]
    num_nodes = len(node_mapping)
    
    print(f"Number of time steps: {num_time_steps}")
    print(f"Number of nodes in network: {num_nodes}")
    print(f"Number of sensors: {len(gdf_sensors)}")
    
    # Debug information about nearest nodes
    print("\nChecking nearest nodes in GeoDataFrame:")
    print("Unique nearest nodes:", gdf_sensors['nearest_node'].unique())
    print("Number of unique nearest nodes:", len(gdf_sensors['nearest_node'].unique()))
    
    # Debug information about node mapping
    print("\nChecking node mapping:")
    print("First few keys in node_mapping:", list(node_mapping.keys())[:5])
    print("Number of keys in node_mapping:", len(node_mapping))
    
    # Initialize full network matrix with zeros
    full_network_matrix = np.zeros((num_time_steps, num_nodes))
    
    # Map sensor data to their corresponding indices in the full network
    sensor_indices = []
    print("\nMapping sensors to nodes:")
    
    # Create a list of all road graph nodes
    road_nodes = list(road_graph.nodes())
    print(f"\nNumber of road nodes: {len(road_nodes)}")
    print("First few road nodes:", road_nodes[:5])
    
    # Map each sensor to its nearest road node
    unmapped_sensors = []
    for sensor_idx, (sensor_id, sensor) in enumerate(gdf_sensors.iterrows()):
        nearest_node = sensor['nearest_node']
        if nearest_node in node_mapping:
            node_idx = node_mapping[nearest_node]
            sensor_indices.append(node_idx)
            
            print(f"Sensor ID: {sensor_id}, Nearest node: {nearest_node}, Node index: {node_idx}, Data index: {sensor_idx}")
            
            if sensor_idx < sensor_data_matrix.shape[1]:
                full_network_matrix[:, node_idx] = sensor_data_matrix[:, sensor_idx]
            else:
                print(f"Warning: Sensor index {sensor_idx} out of bounds for data matrix shape {sensor_data_matrix.shape}")
        else:
            unmapped_sensors.append((sensor_id, nearest_node))
            print(f"Warning: No mapping found for sensor {sensor_id} with nearest node {nearest_node}")
    
    print(f"\nTotal mapped sensors: {len(sensor_indices)}")
    print(f"Total unmapped sensors: {len(unmapped_sensors)}")
    if unmapped_sensors:
        print("\nFirst few unmapped sensors:")
        for sensor_id, node in unmapped_sensors[:5]:
            print(f"Sensor {sensor_id}: nearest node {node}")
    
    return full_network_matrix, sensor_indices

def create_road_features_matrix(road_graph, node_mapping):
    """
    Create a matrix of road features for each node using one-hot encoding for categorical variables
    Args:
        road_graph: NetworkX graph of the road network
        node_mapping: Dictionary mapping node coordinates to indices
    Returns:
        road_features: Matrix of road features [num_nodes, num_features]
    """
    # Define categorical mappings
    kategor_mapping = {
        '1': 0,  # Autostrados
        '2': 1,  # Magistraliniai keliai
        '3': 2,  # Krašto keliai
        '4': 3,  # Rajoniniai keliai
        '5': 4,  # Jungiamieji keliai
        '6': 5,  # Vietiniai keliai
        '7': 6,  # Pagrindinės gatvės
        '8': 7,  # Jungiamosios gatvės
        '9': 8,  # Kitos gatvės
        '10': 9,  # Įvažiavimai į kiemus
        '11': 10,  # Pėsčiųjų gatvės
        '12': 11,  # Pėsčiųjų takai
        '13': 12,  # Aikštės
        '14': 13   # Lėktuvų pakilimo takai
    }
    
    oneway_mapping = {
        'F': 0,  # Vienpusis eismas
        'N': 1,  # Blokuotas eismas
        None: 2  # Default/empty
    }
    
    danga_mapping = {
        'A': 0,  # Asfaltas, asfaltbetonis
        'C': 1,  # Cementbetonis
        'G': 2,  # Gruntas
        'Z': 3,  # Žvyras
        'N': 4   # Neapibrėžta
    }
    
    num_nodes = len(node_mapping)
    # Total features: 14 kategor + 3 oneway + 5 danga = 22 features
    num_features = len(kategor_mapping) + len(oneway_mapping) + len(danga_mapping)
    road_features = np.zeros((num_nodes, num_features))
    
    print("\nProcessing road features for nodes:")
    for node, idx in node_mapping.items():
        # Get all edges connected to this node
        edges = list(road_graph.edges(node, data=True))
        if edges:
            # Initialize feature counters for this node
            kategor_counts = np.zeros(len(kategor_mapping))
            oneway_counts = np.zeros(len(oneway_mapping))
            danga_counts = np.zeros(len(danga_mapping))
            
            for _, _, data in edges:
                # Get and map kategor
                kategor = str(data.get('kategor', '9'))  # Default to '9' if missing
                if kategor in kategor_mapping:
                    kategor_counts[kategor_mapping[kategor]] += 1
                
                # Get and map oneway
                oneway = data.get('oneway')
                if oneway in oneway_mapping:
                    oneway_counts[oneway_mapping[oneway]] += 1
                else:
                    oneway_counts[oneway_mapping[None]] += 1
                
                # Get and map danga
                danga = data.get('danga')
                if danga in danga_mapping:
                    danga_counts[danga_mapping[danga]] += 1
                else:
                    danga_counts[danga_mapping['N']] += 1
            
            # Normalize counts to get probabilities
            if np.sum(kategor_counts) > 0:
                kategor_counts = kategor_counts / np.sum(kategor_counts)
            if np.sum(oneway_counts) > 0:
                oneway_counts = oneway_counts / np.sum(oneway_counts)
            if np.sum(danga_counts) > 0:
                danga_counts = danga_counts / np.sum(danga_counts)
            
            # Combine all features
            road_features[idx] = np.concatenate([kategor_counts, oneway_counts, danga_counts])
            
            if idx < 5:  # Print features for first few nodes
                print(f"\nNode {node}:")
                print(f"  Number of connected edges: {len(edges)}")
                print("  Kategor distribution:", kategor_counts)
                print("  Oneway distribution:", oneway_counts)
                print("  Danga distribution:", danga_counts)
    
    # Print summary statistics
    print("\nRoad features summary:")
    print("Number of nodes with non-zero features:", np.sum(np.any(road_features != 0, axis=1)))
    print("Feature means:", np.mean(road_features, axis=0))
    print("Feature stds:", np.std(road_features, axis=0))
    
    return road_features

def compute_full_network_distance_matrix(road_graph, node_mapping):
    """
    Compute distance matrix for the full network
    Args:
        road_graph: NetworkX graph of the road network
        node_mapping: Dictionary mapping node coordinates to indices
    Returns:
        distance_matrix: Distance matrix [num_nodes, num_nodes]
    """
    num_nodes = len(node_mapping)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, (node1, idx1) in enumerate(node_mapping.items()):
        for j, (node2, idx2) in enumerate(node_mapping.items()):
            if i == j:
                distance_matrix[idx1, idx2] = 0
            else:
                try:
                    shortest_path = nx.shortest_path_length(road_graph, node1, node2, weight="weight")
                    distance_matrix[idx1, idx2] = shortest_path / 1000  # Convert to kilometers
                except nx.NetworkXNoPath:
                    distance_matrix[idx1, idx2] = np.inf
        if i % 100 == 0:
            print(f"Processed {i} nodes")
    
    return distance_matrix

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points in EPSG:3346 coordinates
    Args:
        x1, y1: Coordinates of first point in EPSG:3346
        x2, y2: Coordinates of second point in EPSG:3346
    Returns:
        distance: Distance in meters
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_distance_matrix(node_coords, node_mapping):
    """
    Compute distance matrix using Euclidean distance for EPSG:3346 coordinates
    Args:
        node_coords: Dictionary mapping node indices to (x, y) coordinates in EPSG:3346
        node_mapping: Dictionary mapping node coordinates to indices
    Returns:
        distance_matrix: Distance matrix [num_nodes, num_nodes] in kilometers
    """
    num_nodes = len(node_mapping)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Convert node_mapping to reverse lookup for efficiency
    reverse_mapping = {node: idx for node, idx in node_mapping.items()}
    
    print("Computing Euclidean distances...")
    for i, (node1, idx1) in enumerate(node_mapping.items()):
        if i % 100 == 0:
            print(f"Processing node {i}/{num_nodes}")
        
        x1, y1 = node1
        for node2, idx2 in node_mapping.items():
            if idx1 != idx2:
                x2, y2 = node2
                # Calculate distance in meters and convert to kilometers
                distance = euclidean_distance(x1, y1, x2, y2) / 1000
                distance_matrix[idx1, idx2] = distance
                distance_matrix[idx2, idx1] = distance  # Symmetric

    return distance_matrix

def create_enhanced_adjacency_matrix(distance_matrix, road_features, distance_threshold=0.1, feature_weights=None):
    """
    Create adjacency matrix incorporating both distance and road features
    Args:
        distance_matrix: Distance matrix [num_nodes, num_nodes]
        road_features: Road features matrix [num_nodes, num_features]
        distance_threshold: Distance threshold for creating edges
        feature_weights: Weights for each road feature (default: equal weights)
    Returns:
        adj_matrix: Enhanced adjacency matrix
    """
    if feature_weights is None:
        feature_weights = np.ones(road_features.shape[1]) / road_features.shape[1]
    
    # Create base adjacency matrix based on distance threshold
    adj_matrix = (distance_matrix > 0) & (distance_matrix <= distance_threshold)
    
    # Compute feature similarity matrix
    feature_similarity = np.zeros_like(distance_matrix)
    for i in range(len(road_features)):
        for j in range(len(road_features)):
            if adj_matrix[i, j]:
                # Compute weighted feature similarity
                feature_diff = np.abs(road_features[i] - road_features[j])
                similarity = np.exp(-np.sum(feature_weights * feature_diff))
                feature_similarity[i, j] = similarity
    
    # Combine distance and feature similarity
    distance_weight = 0.7  # Weight for distance component
    feature_weight = 0.3   # Weight for feature similarity component
    
    # Normalize distance matrix
    norm_distance = distance_matrix / distance_matrix.max()
    
    # Create final adjacency matrix
    enhanced_adj = adj_matrix * (
        distance_weight * (1 - norm_distance) +  # Closer nodes get higher weight
        feature_weight * feature_similarity      # Similar features get higher weight
    )
    
    return enhanced_adj
