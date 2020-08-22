import random
import unittest

from evolutionary_grid.EvolutionaryParameterGrid import EAParameterGrid


class TestEvolutionarySearch(unittest.TestCase):
    def test_search(self):
        random.seed(0)
        grid = EAParameterGrid()
        evaluation_time = 0
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            grid.set_fitness(sum(x.values()))
            evaluation_time += 1
        print(evaluation_time)
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)
        self.assertEqual(evaluation_time, 574)

    def test_batch_search(self):
        random.seed(0)
        grid = EAParameterGrid(batch_evaluate=True)
        evaluation_time = 0
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            fitness_list = {}
            for xx in x:
                fitness_list[str(xx)] = (sum(xx.values()))
                evaluation_time += 1
            grid.set_fitness(fitness_list)
        print(evaluation_time)
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)
        self.assertEqual(evaluation_time, 574)

    def test_search_full(self):
        random.seed(0)
        grid = EAParameterGrid(skip_evaluated=False)
        evaluation_time = 0
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            grid.set_fitness(sum(x.values()))
            evaluation_time += 1
        print(evaluation_time)
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)
        self.assertEqual(evaluation_time, 2550)

    def test_batch_full_search(self):
        random.seed(0)
        grid = EAParameterGrid(skip_evaluated=False, batch_evaluate=True)
        evaluation_time = 0
        for x in grid.grid({
            f'x{i}': range(0, 100) for i in range(0, 10)
        }):
            fitness_list = []
            for xx in x:
                fitness_list.append(sum(xx.values()))
                evaluation_time += 1
            grid.set_fitness(fitness_list)
        print(evaluation_time)
        self.assertEqual(grid.halloffame[0].fitness.values[0], 936)
        self.assertEqual(evaluation_time, 2550)


if __name__ == '__main__':
    unittest.main()
