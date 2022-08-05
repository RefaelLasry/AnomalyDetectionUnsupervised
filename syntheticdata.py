import numpy as np


def generate_synthetic_data():
    num_samples = 10
    num_anomalies_samples = 1
    dimension = 5
    x = np.random.normal(loc=0.0, scale=1.0, size=(dimension, num_samples))  # 100000
    anomalies = np.random.normal(loc=10.0, scale=1.0, size=(dimension, num_anomalies_samples))  # 10
    return np.concatenate((x.T, anomalies.T))
