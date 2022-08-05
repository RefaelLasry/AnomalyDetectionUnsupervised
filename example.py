from syntheticdata import generate_synthetic_data
from localoutlierfactorprobabilities import LocalOutlierProbabilities


def main():
    data = generate_synthetic_data()
    local_outlier_probabilities = LocalOutlierProbabilities(phi=0.95, k=2)
    _, local_outlier_probabilities = local_outlier_probabilities.compute_probabilistic_local_outlier_factor(X=data)
    return local_outlier_probabilities


if __name__ == '__main__':
    res = main()
    print(res)
