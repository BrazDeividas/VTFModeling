from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SENSOR_DATA_DIR = DATA_DIR / "sensor_data"

SENSOR_DATA_MATRIX = DATA_DIR / "processed_sensor_data/sensor_data_matrix.pkl"

SENSOR_DISTANCE_MATRIX = DATA_DIR / "processed_sensor_data/sensor_distance_matrix.pkl"

ROAD_GRAPH_DATA = DATA_DIR / "exported_data/road_graph.pkl"

SENSOR_POSITIONS = SENSOR_DATA_DIR / "Sankryzos_koordinates.xlsx"

ROAD_DATA = DATA_DIR / "road_data/keliai_gatves_ExportFeatures1.shp"

SENSOR_GEO_DATA = DATA_DIR / "exported_data/sensor_geo_data.pkl"