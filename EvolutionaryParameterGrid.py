import array
import random

import numpy
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd


def _mutIndividual(individual, up, indpb):
    for i, up in zip(range(len(up)), up):
        if random.random() < indpb:
            individual[i] = random.randint(0, up)
    return individual,


def _cxIndividual(ind1, ind2, indpb):
    for i in range(len(ind1)):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


class EAParameterGrid():
    def __init__(self, generation=50, pop_size=50, cxpb=0.5, mutpb=0.2, tournament_size=10, skip_evaluated=True,
                 halloffame_size=1, batch_evaluate=False,
                 verbose=__debug__):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.verbose = verbose
        self.tournament_size = tournament_size
        self.generation = generation
        self.pop_size = pop_size
        self.current_fitness = 0
        self.skip_evaluated = skip_evaluated
        self.history_dict = dict()
        self.halloffame = tools.HallOfFame(halloffame_size)
        self.batch_evaluate = batch_evaluate

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        self.stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

    def set_fitness(self, current_fitness):
        self.current_fitness = current_fitness

    def _convert_individual_to_dict(self, names, individual):
        final_dict = {}
        for k, v in zip(names, list(individual)):
            final_dict[k] = self.parameter_grid[k][v]
        return final_dict

    def _history_check(self, individual):
        if tuple(individual) in self.history_dict:
            return True
        else:
            return False

    def grid(self, parameter_grid):
        self.parameter_grid = parameter_grid
        names = parameter_grid.keys()
        maxints = [len(possible_values) - 1 for possible_values in parameter_grid.values()]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Individual generator
        def individual_generator():
            return creator.Individual([random.randint(0, x) for x in maxints])

        # Structure initializers
        toolbox.register("population", tools.initRepeat, list, individual_generator)

        toolbox.register("mate", _cxIndividual, indpb=self.cxpb)
        toolbox.register("mutate", _mutIndividual, up=maxints, indpb=self.mutpb)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        population = toolbox.population(n=self.pop_size)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])

        yield from self.population_evaluation(names, population)

        if self.halloffame is not None:
            self.halloffame.update(population)

        record = self.stats.compile(population) if self.stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if self.verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self.generation + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluate the individuals with an invalid fitness
            yield from self.population_evaluation(names, offspring)

            # Update the hall of fame with the generated individuals
            if self.halloffame is not None:
                self.halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, nevals=len(offspring), **record)
            if self.verbose:
                print(logbook.stream)

        return population, logbook

    def population_evaluation(self, names, population):
        if self.batch_evaluate:
            batch_list = []
            for ind in population:
                if self.skip_evaluated and self._history_check(ind):
                    ind.fitness.values = self.history_dict[tuple(ind)]
                    continue
                else:
                    batch_list.append(ind)
            yield [self._convert_individual_to_dict(names, ind) for ind in batch_list]
            for k,v in zip(batch_list,self.current_fitness):
                k.fitness.values = v,
        else:
            for ind in population:
                if self.skip_evaluated and self._history_check(ind):
                    ind.fitness.values = self.history_dict[tuple(ind)]
                    continue
                yield self._convert_individual_to_dict(names, ind)
                ind.fitness.values = self.current_fitness,
                self.history_dict[tuple(ind)] = ind.fitness.values
