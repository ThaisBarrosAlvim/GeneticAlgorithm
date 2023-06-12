import datetime
import os.path
import random
import statistics
import time
from collections import defaultdict
from functools import cached_property
from typing import Optional, Literal

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def hour_to_min(hour: str):
    return int(hour.split(':')[0]) * 60 + int(hour.split(':')[1])


def min_to_hour(minutes: int):
    return str(minutes // 60) + ':' + str(minutes % 60) if minutes % 60 > 9 else str(minutes // 60) + ':0' + str(
        minutes % 60)


class Flight:
    def __init__(self, index: int, dep_airport: str, arr_airport: str, dep_time: int, arr_time: int, price: int):
        self.index = index
        self.dep_airport = dep_airport
        self.arr_airport = arr_airport
        self.dep_time = dep_time
        self.arr_time = arr_time
        self.price = price

    def dep_time_show(self):
        return min_to_hour(self.dep_time)

    def arr_time_show(self):
        return min_to_hour(self.arr_time)

    def get_group(self):
        return self.dep_airport + '|' + self.arr_airport

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.index == other.index

    def __str__(self):
        return f'Flight({self.index})'

    def full_str(self, time_sec: bool = True):
        return f'Flight({self.index}): {self.dep_airport}|{self.arr_airport} ' \
               f'AT={self.arr_time if time_sec else self.arr_time_show()} ' \
               f'DT={self.dep_time if time_sec else self.dep_time_show()}' \
               f' P={self.price}'

    def full_str_def(self, time_sec: bool = True):
        # FCO,LIS,6:19,8:13,239
        if time_sec:
            return f'{self.dep_airport},{self.arr_airport},{self.dep_time},{self.arr_time},{self.price}'
        else:
            return f'{self.dep_airport},{self.arr_airport},{self.dep_time_show()},{self.arr_time_show()},{self.price}'

    def __repr__(self):
        return str(self)

    @classmethod
    def from_string(cls, index, s):
        # FCO,LIS,6:19,8:13,239
        s = s.split(',')
        return cls(index, s[0], s[1], hour_to_min(s[2]), hour_to_min(s[3]), int(s[4]))


class Individual:

    def __init__(self, flights: Optional[list] = None, itype: Literal['Parcial', 'Complete'] = 'Parcial'):
        if flights is None:
            flights = []
        self._flights = flights
        self.itype = itype

    def __str__(self):
        return f'Individual({self.fitness})'

    def full_str(self, time_sec: bool = True):
        div = '\n\t'
        return f'Individual({self.fitness}): \n\t{div.join(f.full_str_def(time_sec) for f in self._flights)}\n'

    def __hash__(self):
        return hash(tuple(self._flights))

    def __eq__(self, other):
        return self._flights == other._flights

    @property
    def flights(self) -> list[Flight]:
        return self._flights

    def insert_flight(self, value: Flight, index: int = None):
        if index is None:
            self._flights.append(value)
        else:
            self._flights[index] = value

        # delete cached fitness if it exists
        try:
            del self.fitness
        except:
            pass

    @flights.setter
    def flights(self, value):
        self._flights = value
        try:
            del self.fitness
        except:
            pass

    @cached_property
    def fitness(self):
        if self.itype == 'Parcial':
            if self.flights[0].dep_airport == GeneticAlgorithmMin.TARGET_AIRPORT:
                time_field = 'dep_time'
            elif self.flights[0].arr_airport == GeneticAlgorithmMin.TARGET_AIRPORT:
                time_field = 'arr_time'
            else:
                raise ValueError('All flights must be in target airport')

            # calculate the max time from/to target airport
            greatest_time = getattr(self.flights[0], time_field)
            for flight in self.flights[1:]:  # type: Flight
                if getattr(flight, time_field) > greatest_time:
                    greatest_time = getattr(flight, time_field)

            fitness = 0
            # sum all differences between max and min and prices
            for flight in self.flights:
                fitness += flight.price * GeneticAlgorithmMin.PRICE_WEIGHT
                fitness += greatest_time - getattr(flight, time_field)
        else:
            # calculate the max arrival time to target airport
            greatest_arr_time = self.flights[0].arr_time
            for i in self.flights[1:6]:  # type: Flight
                if i.arr_time > greatest_arr_time:
                    greatest_arr_time = i.arr_time

            # calculate the max departure time from target airport
            greatest_dep_time = self.flights[6].dep_time
            for i in self.flights[7:]:  # type: Flight
                if i.dep_time > greatest_dep_time:
                    greatest_dep_time = i.dep_time

            fitness = 0
            # sum all differences between max and min and prices
            for i in range(len(self.flights)):
                fitness += self.flights[i].price * GeneticAlgorithmMin.PRICE_WEIGHT
                if i < 6:
                    if self.flights[i].arr_time != greatest_arr_time:
                        fitness += greatest_arr_time - self.flights[i].arr_time
                else:
                    if self.flights[i].dep_time != greatest_dep_time:
                        fitness += greatest_dep_time - self.flights[i].dep_time

        return fitness

    def copy(self):
        return Individual(self.flights.copy())

    def __repr__(self):
        return str(self)

    @classmethod
    def create_individual(cls, flights: dict, target_airport: str, dep_from_target: bool = True):

        # add random flights in order of arrive in target_airport and then depart from target_airport
        flights_keys = []
        for f in sorted(flights, key=lambda x: target_airport == x[4:], reverse=True):
            if (f[:3] == target_airport and dep_from_target) \
                    or (f[4:] == target_airport and not dep_from_target):
                flights_keys.append(f)

        indvidual = cls()
        for k in flights_keys:
            indvidual.insert_flight(random.choice(flights[k]))

        return indvidual


class GeneticAlgorithmMin:
    TARGET_AIRPORT = 'FCO'
    PRICE_WEIGHT = 1

    def __init__(self):
        self.population = []


    def run(self, population_size: int, generation_size: int, crossover_rate: float, mutation_rate: float,
            tournament_size: int, tournament_k: float, flights: dict, dep_from_target: bool = True,
            genes_to_mutate: Optional[int] = None, run_data: list = None, individuals: list = None):
        # create the initial population
        self.population.clear()
        for i in range(population_size):
            self.population.append(Individual.create_individual(flights, self.TARGET_AIRPORT,
                                                                dep_from_target=dep_from_target))

        best = self.population[0]
        data = {'avg_fitness': [], 'best': [], 'worst': []}
        for i in range(generation_size):
            # Elitism: copy the best individual of the previous generation
            best = best.copy()

            # Selection by tournament
            population_to_crossover = self._tournament_selection(tournament_size, tournament_k)
            # Crossover
            self._crossover_population(population_to_crossover, crossover_rate)

            # Mutation
            if genes_to_mutate is None:
                # mutate all genes
                for ip in range(len(self.population)):
                    self.population[ip] = self._bit_flip_mutation_all_genes(self.population[ip], flights, mutation_rate)
            else:
                # mutate only a certain number of genes
                for ip in range(len(self.population)):
                    if random.random() < mutation_rate:
                        self.population[ip] = self._bit_flip_mutation(self.population[ip], flights, genes_to_mutate)

            data['avg_fitness'].append(sum([x.fitness for x in self.population]) / len(self.population))
            worse_individual = max(self.population, key=lambda x: x.fitness)
            data['worst'].append(worse_individual.fitness)

            # Elitism: replace the worst individual of the generation with the best of the previous generation
            if worse_individual.fitness > best.fitness:
                worse_index = self.population.index(worse_individual)
                self.population[worse_index] = best

            # Get the new best individual of the generation and add it to the data
            best = min(self.population, key=lambda x: x.fitness)
            data['best'].append(best.fitness)

        best = min(self.population, key=lambda x: x.fitness)
        if individuals is not None and run_data is not None:
            individuals.append(best)
            run_data.append(data)
        else:
            return best, data

    def _tournament_selection(self, size: int, k: float = 0.75):
        new_population = []
        for i in range(len(self.population)):
            tournament = random.sample(self.population, size)
            # the goal is to minimize the fitness
            if random.random() < k:
                winner = min(tournament, key=lambda x: x.fitness)
            else:
                winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
        return new_population

    def _uniform_crossover(self, parent1: Individual, parent2: Individual):
        uniform_mask = [random.choice([True, False]) for _ in range(len(parent1.flights))]
        child1_flights, child2_flights = [], []
        for i in range(len(parent1.flights)):
            if uniform_mask[i]:
                child1_flights.append(parent1.flights[i])
                child2_flights.append(parent2.flights[i])
            else:
                child1_flights.append(parent2.flights[i])
                child2_flights.append(parent1.flights[i])
        child1 = Individual(child1_flights)
        child2 = Individual(child2_flights)
        return child1, child2

    def _crossover_population(self, population: list, crossover_rate: float):
        new_population = []
        p_size = len(population)
        population = population.copy()
        while len(new_population) < p_size:
            if len(population) > 1:
                parent1, parent2 = random.sample(population, 2)
                population.remove(parent1)
                population.remove(parent2)
                if random.random() < crossover_rate:
                    child1, child2 = self._uniform_crossover(parent1, parent2)
                    new_population.append(child1)
                    new_population.append(child2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            else:
                new_population.append(population[0])

        self.population = new_population

    def _bit_flip_mutation_all_genes(self, individual: Individual, flights: dict, mutation_rate: float = 0.05):
        for i in range(len(individual.flights)):
            if random.random() < mutation_rate:
                possible_flights = flights[individual.flights[i].get_group()]
                # remove the flight that is already in the individual from the possible flights to mutate
                possible_flights = [f for f in possible_flights if f != individual.flights[i]]
                new_flight = random.choice(possible_flights)
                individual.insert_flight(new_flight, i)
        return individual

    def _bit_flip_mutation(self, individual: Individual, flights: dict, genes_to_mutate: int = 1):
        for i in range(genes_to_mutate):
            selected_to_mutate = random.randint(0, len(individual.flights) - 1)
            possible_flights = flights[individual.flights[selected_to_mutate].get_group()]
            # remove the flight that is already in the individual from the possible flights to mutate
            possible_flights = [f for f in possible_flights if f != individual.flights[selected_to_mutate]]
            new_flight = random.choice(possible_flights)
            individual.insert_flight(new_flight, selected_to_mutate)
        return individual


def create_flight_segment_to_flights(data_path: str) -> dict[str, list[Flight]]:
    """
     Create a dictionary that maps a flight segment to all the flights that can be used for that segment
     :returns format {'{dep_airport}|{arr_airport}': [flight1, flight2, ...]}
    """
    flights = defaultdict(list)
    df = pd.read_csv(data_path, sep=',', header=None)
    for index, row in enumerate(df.iterrows()):
        # FCO,LIS,6:19,8:13,239
        flights[f'{row[1][0]}|{row[1][1]}'].append(Flight.from_string(index, ','.join(row[1].astype(str))))
    return flights


def plot_data(data: list, save_dir: str, context_title: str = '', context_data: str = '', multiple_plots=False,
              ylabel='Fitness', xlabel='Generation', ) -> None:
    if multiple_plots:
        for d in data:
            plt.plot(d)
    else:
        plt.plot(data)
    plt.suptitle(context_title)
    plt.title(context_data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # set x axis to integer
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if multiple_plots:
        min_data = min(min(d) for d in data)
    else:
        min_data = min(data)
    plt.axhline(min_data, color='r', linestyle='--')
    plt.text(0, min_data + 10, f'Best fitness: {min_data}')
    # prevent same name plots, adding a number at the end
    if os.path.exists(os.path.join(save_dir, f'{context_title}.png')):
        i = 1
        while os.path.exists(os.path.join(save_dir, f'{context_title}_{i}.png')):
            i += 1
        plt.savefig(os.path.join(save_dir, f'{context_title}_{i}.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_dir, f'{context_title}.png'), bbox_inches='tight')
    plt.show()


def test_runner(tests: int, data_path: str, ):
    # Test parameters
    date = datetime.datetime.today().strftime("%d-%m_%H-%M-%S")
    reports_dir = os.path.join('reports', date)
    plot_dir = os.path.join(reports_dir, 'plots')
    make_reports = True  # create file with all test results or print
    if make_reports:
        # check if dir exists
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    # plot parameters
    plot_all = True  # will plot all best individuals of each run
    plot_all_best_fitness = True
    plot_all_fitness_through_generations = True
    plot_all_avg_fitness_through_generations = True

    # GeneticAlgorithmMin parameters
    GeneticAlgorithmMin.TARGET_AIRPORT = 'FCO'
    GeneticAlgorithmMin.PRICE_WEIGHT = 1

    # Genetic algorithm parameters
    population_size = 1500
    crossover_rate = 0.8
    mutation_rate = 0.05
    genes_to_mutate = None  # None -> all genes can mutate
    generation_size = 15
    tournament_size = 15
    tournament_k = 0.61  # higher k -> higher chance of selecting the best individual in the tournament

    # Read file
    flights = create_flight_segment_to_flights(data_path)

    start = time.time()
    individuals = []  # list of best individuals of each run
    runs_data = []  # list of data of each run (to plot info of each run)
    random.seed()

    genetic_algorithm = GeneticAlgorithmMin()

    for i in range(tests):
        tini = time.time()
        # first part arrive in target airport
        best1, data1 = genetic_algorithm.run(population_size, generation_size, crossover_rate, mutation_rate,
                                             tournament_size, tournament_k, flights, dep_from_target=False,
                                             genes_to_mutate=genes_to_mutate)

        # second part departure from target airport
        best2, data2 = genetic_algorithm.run(population_size, generation_size, crossover_rate, mutation_rate,
                                             tournament_size, tournament_k, flights, dep_from_target=True,
                                             genes_to_mutate=genes_to_mutate)


        # Merge both results, individual and data
        best = Individual(best1.flights + best2.flights, itype='Complete')
        data = {dk: [v1 + v2 for v1, v2 in zip(data1[dk], data2[dk])] for dk in data1.keys()}

        print(f'Test({i + 1}): Best fitness: {best.fitness}, takes: {(time.time() - tini):.3f} seconds\n')

        if plot_all:
            plot_data(data['best'], plot_dir, f'Test({i + 1}) Best Fitness ({best.fitness})', f'Population size: {population_size},'
                                                                             f' Crossover rate: {crossover_rate},'
                                                                             f' Mutation rate: {mutation_rate}',
                      ylabel='Fitness', xlabel='Generation')

        individuals.append(best)
        runs_data.append(data)

    end = time.time()
    runs_results = [individual.fitness for individual in individuals]
    if tests > 1:
        # plot data of all runs
        if plot_all_best_fitness:
            plot_data(runs_results, plot_dir, f'All Tests Best Fitness', f'Population size: {population_size},'
                                                                         f' Crossover rate: {crossover_rate},'
                                                                         f' Mutation rate: {mutation_rate}',
                      ylabel='Fitness', xlabel='Test number')
        # plot all in one graph
        if plot_all_fitness_through_generations:
            plot_data([rd['best'] for rd in runs_data], plot_dir,
                      f'All Tests Best Fitness Through Generations', f'Population size: {population_size},'
                                                                     f' Crossover rate: {crossover_rate},'
                                                                     f' Mutation rate: {mutation_rate}',
                      multiple_plots=True,
                      ylabel='Fitness', xlabel='Generation')

        if plot_all_avg_fitness_through_generations:
            plot_data([rd['avg_fitness'] for rd in runs_data], plot_dir,
                      f'All Tests Average Fitness Through Generations',
                      f'Population size: {population_size},'
                      f' Crossover rate: {crossover_rate},'
                      f' Mutation rate: {mutation_rate}', ylabel='Average Fitness', xlabel='Generation',
                      multiple_plots=True)

        # Write results to file
        avg = sum(runs_results) / len(runs_results)
        best = min(runs_results)
        worst = max(runs_results)
        stdev = statistics.stdev(runs_results)
        variance = statistics.variance(runs_results)
        median = statistics.median(runs_results)
        tdelta = end - start
        avg_time = tdelta / tests
        multiple_tests_txt = (f'\nIt takes {tdelta:.3f} seconds to run {tests} times'
                              f'\nAverage time: {avg_time:.3f} seconds'
                              f'\nAverage best fitness: {avg}'
                              f'\nBest fitness: {best}'
                              f'\nWorst fitness: {worst}'
                              f'\nStandard deviation: {stdev} (how far from the mean)'
                              f'\nVariance: {variance} (square of standard deviation)'
                              f'\nMedian: {median} (middle value)')
    else:
        multiple_tests_txt = ''
        stdev = avg = 0
        best = runs_results[0]

    individuals_txt = "\n\n".join([f"Individual {i + 1}: {individuals[i].full_str(time_sec=False)}"
                                   for i in range(len(individuals))])
    txt = (f'Date: {date}\n'
           f'Configs:'
           f'\n\ttests={tests}'
           f'\n\tpopulation_size={population_size}'
           f'\n\tgeneration_size={generation_size}'
           f'\n\tcrossover_rate={crossover_rate}'
           f'\n\tmutation_rate={mutation_rate}'
           f'\n\tgenes_to_mutate={genes_to_mutate}'

           f'\nTournament selection:'
           f'\n\ttournament_size={tournament_size}'
           f'\n\ttournament_k={tournament_k}'

           f'\n\n*{"--*" * 6}  Results *{"--*" * 6}'
           f'{multiple_tests_txt}'
           f'\n\nBests: {runs_results}'
           f'\n\nIndividuals: {individuals_txt}')
    print(txt)

    # write to file
    if make_reports:
        file_name = f'{date}_best{best}' \
                    f'_var{stdev:.2f}' \
                    f'_avg{avg:.2f}' \
                    f'_ga_p{population_size}' \
                    f'_g{generation_size}' \
                    f'_cr{crossover_rate}' \
                    f'_mr{mutation_rate}' \
                    f'_tz{tournament_size}' \
                    f'_gtm{genes_to_mutate if genes_to_mutate else "N"}' \
                    f'_tk{tournament_k:.2f}.txt'
        with open(os.path.join(reports_dir, file_name), 'w') as file:
            file.write(txt)


if __name__ == '__main__':
    qtd_tests = 30
    flights_path = 'flights.txt'
    test_runner(qtd_tests, flights_path)
