from shapely.geometry import Point
import pandas as pd

def zscore(x, mean, std):
    return (x - mean) / std

def inverse_zscore(x, mean, std):
  return x * std + mean

def smooth_data(normalized_data, span = 5):
    return pd.DataFrame(normalized_data).ewm(span=span, adjust=False).mean().to_numpy()

def find_nearest_node(graph, point):
    return min(graph.nodes, key=lambda x: Point(x).distance(point))