import random

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from evolutionary_grid.EvolutionaryParameterGrid import EAParameterGrid

random.seed(0)
np.random.seed(0)

X, y = load_boston(return_X_y=True)
param_grid = {
    "max_depth": np.random.randint(1, (X.shape[1] * .85), 20),
    "max_features": np.random.randint(1, X.shape[1], 20),
    "min_samples_leaf": [2, 3, 4, 5, 6],
    "criterion": ["mse", "mae"],
}

grid = EAParameterGrid()
for param in grid.grid(param_grid):
    tree = DecisionTreeRegressor(**param, random_state=0)
    score = cross_val_score(tree, X, y, cv=5)
    grid.set_fitness(np.mean(score))
