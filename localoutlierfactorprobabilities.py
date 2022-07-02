import numpy as np
from scipy import special
from scipy.spatial import distance_matrix


class LocalOutlierProbabilities:
    def __init__(self, phi, k):
        self.phi = phi
        self.k = k

    def compute_lambda_function(self):
        return np.sqrt(2) * special.erfinv(self.phi)

    def compute_distance_function(self, point_a, point_b):
        """
        Assuming Euclidean distance
        """
        return np.linalg.norm(point_a - point_b)

    def create_subset_of_set_of_points(self, point, set_of_points):
        m, n = np.meshgrid(point, set_of_points)
        distances = distance_matrix(n, m)
        indices = np.argpartition(distances[:, 0], -self.k)[-self.k:]
        subset_of_set_of_points = np.take(set_of_points, indices)
        return subset_of_set_of_points

    def compute_standard_distance(self, point, set_of_points):
        """
        sigma on the article
        """
        subset_of_set_of_points = self.create_subset_of_set_of_points(set_of_points=set_of_points, point=point)
        distances = []
        for i in subset_of_set_of_points:
            distance = self.compute_distance_function(point_a=point, point_b=i)
            distances.append(distance)
        squares_of_distances = np.power(distances, 2)
        sum_squares_of_distances = sum(squares_of_distances)
        norm_of_set = np.linalg.norm(subset_of_set_of_points)
        return np.sqrt(sum_squares_of_distances/norm_of_set)

    def compute_probabilistic_set_distance(self, point, set_of_points):
        lambda_value = self.compute_lambda_function()
        standard_distance = self.compute_standard_distance(point=point, set_of_points=set_of_points)
        return lambda_value * standard_distance

    def compute_expectation_of_probabilistic_set_distance(self, set_of_points):
        values = []
        for i in set_of_points:
            value = self.compute_probabilistic_set_distance(point=i, set_of_points=set_of_points)
            values.append(value)
        return np.mean(values)

    def compute_local_outlier_factor(self, X):
        y = []
        expectation_of_probabilistic_set_distance = self.compute_expectation_of_probabilistic_set_distance(set_of_points=X)
        for i in X:
            probabilistic_set_distance = self.compute_probabilistic_set_distance(point=i, set_of_points=X)
            value = (probabilistic_set_distance / expectation_of_probabilistic_set_distance) - 1
            y.append(value)
        return np.array(y)

    def compute_n_plof(self, X):
        lambda_value = self.compute_lambda_function()
        local_outlier_factor = self.compute_local_outlier_factor(X)
        local_outlier_factor_squares = local_outlier_factor ** 2
        mean_local_outlier_factor_squares = np.mean(local_outlier_factor_squares)
        n_plof = lambda_value * np.sqrt(mean_local_outlier_factor_squares)
        return n_plof

    def compute_probabilistic_local_outlier_factor(self, X):
        local_outlier_factor = self.compute_local_outlier_factor(X=X)
        n_plof = self.compute_n_plof(X=X)
        values = local_outlier_factor / (np.sqrt(2)*n_plof)
        prob_values = special.erf(values)
        local_outlier_probabilities = np.maximum(0, prob_values)
        return X, local_outlier_probabilities

