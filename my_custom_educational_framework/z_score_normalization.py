def z_score_normalization(X):
    means = X.mean(0)
    standard_deviations = X.std(0)
    X = X - means
    X = X / standard_deviations

    return (X, means, standard_deviations)