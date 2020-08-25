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
        """
        EA parameter grid initialization

        :param generation: Max number of generations to be evolved
        :param pop_size: Population size of genetic algorithm
        :param cxpb: Probability of gene mutation in chromosome
        :param mutpb: Probability of gene swap between two chromosomes
        :param tournament_size: Size of tournament for selection stage of genetic algorithm
        :param skip_evaluated: Skip repetitive individuals
        :param halloffame_size: Max number of history best individuals to be saved
        :param batch_evaluate: Evaluate individuals in parallel
        :param verbose: Controls the verbosity: the higher, the more messages.
        """
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.verbose = verbose
        self.tournament_size = tournament_size
        self.generation = generation
        self.pop_size = pop_size
        self.current_fitness = None
        self.skip_evaluated = skip_evaluated
        self.history_dict = dict()
        self.halloffame = tools.HallOfFame(halloffame_size)
        self.batch_evaluate = batch_evaluate
        self.names = dict()

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        self.stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    def set_fitness(self, current_fitness):
        self.current_fitness = current_fitness

    def _convert_individual_to_dict(self, individual):
        final_dict = {}
        for k, v in zip(self.names, list(individual)):
            final_dict[k] = self.parameter_grid[k][v]
        return final_dict

    def _history_check(self, individual):
        if tuple(individual) in self.history_dict:
            return True
        else:
            return False

    def best_individuals(self):
        return [self._convert_individual_to_dict(hof) for hof in self.halloffame]

    def grid(self, parameter_grid):
        self.parameter_grid = parameter_grid
        self.names = parameter_grid.keys()
        maxints = [len(possible_values) - 1 for possible_values in parameter_grid.values()]

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

        yield from self._population_evaluation(population)

        if self.halloffame is not None:
            self.halloffame.update(population)

        record = self.stats.compile(population) if self.stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if self.verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self.generation + 1):
            if self.batch_evaluate and self.skip_evaluated:
                new_offsprings = []
                offspring_set = set()
                while len(new_offsprings) < len(population):
                    # Select the next generation individuals
                    offspring = toolbox.select(population, len(population))
                    # Vary the pool of individuals
                    offspring = varAnd(offspring, toolbox, self.cxpb, self.mutpb)
                    for o in offspring:
                        if len(new_offsprings) >= len(population):
                            break
                        if tuple(o) in offspring_set or tuple(o) in self.history_dict:
                            continue
                        offspring_set.add(tuple(o))
                        new_offsprings.append(o)
                offspring = new_offsprings
            else:
                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))
                # Vary the pool of individuals
                offspring = varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluate the individuals with an invalid fitness
            yield from self._population_evaluation(offspring)

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

    def _population_evaluation(self, population):
        if self.batch_evaluate:
            batch_list = []
            batch_set = set()
            for ind in population:
                if self.skip_evaluated and self._history_check(ind):
                    ind.fitness.values = self.history_dict[tuple(ind)]
                    continue
                else:
                    batch_list.append(ind)
                    batch_set.add(tuple(ind))
            if self.skip_evaluated:
                yield [self._convert_individual_to_dict(ind) for ind in batch_set]
            else:
                yield [self._convert_individual_to_dict(ind) for ind in batch_list]
            for index, k in enumerate(batch_list):
                k.fitness.values = (self.current_fitness[str(self._convert_individual_to_dict(k))],) \
                    if self.skip_evaluated else (self.current_fitness[index],)
                self.history_dict[tuple(k)] = k.fitness.values
        else:
            for ind in population:
                if self.skip_evaluated and self._history_check(ind):
                    ind.fitness.values = self.history_dict[tuple(ind)]
                    continue
                yield self._convert_individual_to_dict(ind)
                ind.fitness.values = self.current_fitness,
                self.history_dict[tuple(ind)] = ind.fitness.values
