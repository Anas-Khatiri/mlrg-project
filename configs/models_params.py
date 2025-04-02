RANDOM_STATE = 0

# Paramètres pour Lasso
LASSO_PARAMS = {
    "alpha": [0.0001, 0.001, 0.01, 0.1],  # Forces de régularisation
    "fit_intercept": [True, False],  # Si l'on ajuste l'ordonnée à l'origine
    "max_iter": [1000, 5000],  # Nombre max d'itérations pour la convergence
    "tol": [1e-4, 1e-3],  # Tolérance pour l'optimisation
    "random_state": RANDOM_STATE,  # Reproductibilité
    "selection": ["cyclic", "random"]  # Méthode de mise à jour des coefficients
}

# Paramètres pour XGBoost (existant, à conserver)
XGB_PARAMS = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "random_state": RANDOM_STATE
}

# Paramètres pour MLP (existant, à conserver)
MLP_PARAMS = {
    "hidden_layer_sizes": [(64,), (128, 64)],
    "activation": ["relu"],
    "early_stopping": [True],
    "random_state": RANDOM_STATE
}