RANDOM_STATE = 0

# Paramètres pour Lasso
LASSO_PARAMS = {
    "alpha": [0.0001, 0.001, 0.01, 0.1],  # Forces de régularisation
    "max_iter": [100, 500, 1000],  # Nombre max d'itérations pour la convergence
    # "selection": ["random"]  # Méthode de mise à jour des coefficients
}

# Paramètres pour XGBoost (existant, à conserver)
XGB_PARAMS = {
    "n_estimators": [500, 1000, 2000],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [3, 5, 10],
}

# Paramètres pour MLP (existant, à conserver)
MLP_PARAMS = {
    "hidden_layer_sizes": [(64,), (128, 64)],
    "activation": ["relu"],
}
