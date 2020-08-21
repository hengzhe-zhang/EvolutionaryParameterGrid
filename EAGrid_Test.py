import random
import unittest

from EvolutionaryParameterGrid import EAParameterGrid


class TestEvolutionarySearch(unittest.TestCase):
    def test_search(self):
        random.seed(0)
        grid = EAParameterGrid()
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            grid.set_fitness(sum(x.values()))
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)

    def test_batch_search(self):
        random.seed(0)
        grid = EAParameterGrid(batch_evaluate=True)
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            fitness_list = []
            for xx in x:
                fitness_list.append(sum(xx.values()))
            grid.set_fitness(fitness_list)
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)


if __name__ == '__main__':
    unittest.main()
