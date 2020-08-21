# Evolutionary Parameter Grid
![](https://img.shields.io/pypi/v/EvolutionaryParameterGrid)

This package is a general combinatorial optimization problem solver. Just as the combinatorial optimization solver in the "Scikit-learn" package, this package can also be used in parameter searching, feature selection, or even neural architecture search, as long as the parameter grid is well designed. However, in contrast to the parameter grid which uses the exhaustive search method, this package searches the search space based on an intelligent algorithm, genetic algorithm. Based on this algorithm, the search space can be searched efficiently. 
 
There are a lot of hyper-parameter tuning packages based on the evolutionary algorithm. However, few of them can be used as a general combinatorial optimization solver. Moreover, most of those algorithms do not consider eliminating repetitive proposals, which may lead to cost a lot of computation resources on the same search space. Therefore, in this package, we propose a general solver based on the idea of parameter grid, with the use of a search history memory module, to solve those problems.

[Here](https://github.com/zhenlingcn/EvolutionaryParameterGrid/blob/master/EAGrid_ParameterTuning.ipynb) is an example of the evolutionary parameter grid on the hyper-parameter search algorithm.

## Usage Example
Example of usage:
```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from EvolutionaryParameterGrid import EAParameterGrid

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

```
Output:
```text
gen	nevals	avg     	std     	min     	max     
0  	50    	0.172936	0.427611	-2.23543	0.521039
1  	50    	0.397736	0.11816 	0.0286067	0.522334
2  	50    	0.476128	0.101181	0.156791 	0.522334
3  	50    	0.48994 	0.0817068	0.240743 	0.522334
4  	50    	0.487414	0.0961236	0.133501 	0.522334
5  	50    	0.489379	0.123043 	-0.290536	0.522334
6  	50    	0.498634	0.0721583	0.189493 	0.522334
7  	50    	0.495893	0.090845 	0.0234607	0.522334
8  	50    	0.484376	0.0968043	0.11605  	0.522334
9  	50    	0.507147	0.0487307	0.268768 	0.522334
10 	50    	0.490803	0.115769 	-0.104581	0.522334
```