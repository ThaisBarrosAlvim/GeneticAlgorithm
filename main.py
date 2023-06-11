import datetime
import os.path
import random
import statistics
import threading
import time
from collections import defaultdict
from functools import cached_property
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt


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
    PRICE_WEIGHT = 1

    def __init__(self, flights: Optional[list] = None):
        if flights is None:
            flights = []
        self._flights = flights

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
    def flights(self):
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
            fitness += self.flights[i].price * self.PRICE_WEIGHT
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


def read_file(path):
    flights = defaultdict(list)
    df = pd.read_csv(path, sep=',', header=None)
    for index, row in enumerate(df.iterrows()):
        # FCO,LIS,6:19,8:13,239
        flights[f'{row[1][0]}|{row[1][1]}'].append(Flight.from_string(index, ','.join(row[1].astype(str))))
    return flights


def create_individual(flights: dict, target_airport: str):
    organism = Individual()
    # add random flights in order of arrive in TARGET_AIRPORT and then depart from TARGET_AIRPORT
    for k in sorted(flights, key=lambda x: target_airport == x[4:], reverse=True):
        organism.insert_flight(random.choice(flights[k]))

    return organism


def tournament_selection(population: list, size: int, k: float = 0.75):
    new_population = []
    for i in range(len(population)):
        tournament = random.sample(population, size)
        # the goal is to minimize the fitness
        if random.random() < k:
            winner = min(tournament, key=lambda x: x.fitness)
        else:
            winner = max(tournament, key=lambda x: x.fitness)
        new_population.append(winner)
    return new_population


def uniform_crossover(parent1: Individual, parent2: Individual):
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


def crossover_population(population: list, crossover_rate: float):
    new_population = []
    p_size = len(population)
    population = population.copy()
    while len(new_population) < p_size:
        if len(population) > 1:
            parent1, parent2 = random.sample(population, 2)
            population.remove(parent1)
            population.remove(parent2)
            if random.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        else:
            new_population.append(population[0])

    return new_population


def bit_flip_mutation_all_genes(individual: Individual, flights: dict, mutation_rate: float = 0.05):
    for i in range(len(individual.flights)):
        if random.random() < mutation_rate:
            possible_flights = flights[individual.flights[i].get_group()]
            # remove the flight that is already in the individual from the possible flights to mutate
            possible_flights = [f for f in possible_flights if f != individual.flights[i]]
            new_flight = random.choice(possible_flights)
            individual.insert_flight(new_flight, i)
    return individual


def bit_flip_mutation(individual: Individual, flights: dict, genes_to_mutate: int = 1):
    for i in range(genes_to_mutate):
        selected_to_mutate = random.randint(0, len(individual.flights) - 1)
        possible_flights = flights[individual.flights[selected_to_mutate].get_group()]
        # remove the flight that is already in the individual from the possible flights to mutate
        possible_flights = [f for f in possible_flights if f != individual.flights[selected_to_mutate]]
        new_flight = random.choice(possible_flights)
        individual.insert_flight(new_flight, selected_to_mutate)
    return individual


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


def genetic_algorithm(population_size: int, generation_size: int, crossover_rate: float, mutation_rate: float,
                      tournament_size: int, tournament_k: float, flights: dict, target_airport: str,
                      genes_to_mutate: Optional[int] = None, run_data: list = None, individuals: list = None):
    population = []

    for i in range(population_size):
        population.append(create_individual(flights, target_airport))

    best = population[0]
    data = {'avg_fitness': [], 'best': [], 'worst': []}
    for i in range(generation_size):
        # Elitism: copy the best individual of the previous generation
        best = best.copy()

        # Selection by tournament
        population_to_crossover = tournament_selection(population, tournament_size, tournament_k)
        # Crossover
        population = crossover_population(population_to_crossover, crossover_rate)

        # Mutation
        if genes_to_mutate is None:
            for ip in range(len(population)):
                population[ip] = bit_flip_mutation_all_genes(population[ip], flights, mutation_rate)
        else:
            for ip in range(len(population)):
                if random.random() < mutation_rate:
                    population[ip] = bit_flip_mutation(population[ip], flights, genes_to_mutate)

        data['avg_fitness'].append(sum([x.fitness for x in population]) / len(population))
        worse_individual = max(population, key=lambda x: x.fitness)
        data['worst'].append(worse_individual.fitness)

        # Elitism: replace the worst individual of the generation with the best of the previous generation
        if worse_individual.fitness > best.fitness:
            worse_index = population.index(worse_individual)
            population[worse_index] = best

        # Get the new best individual of the generation and add it to the data
        best = min(population, key=lambda x: x.fitness)
        data['best'].append(best.fitness)

    best = min(population, key=lambda x: x.fitness)
    if individuals is not None and run_data is not None:
        individuals.append(best)
        run_data.append(data)
    else:
        return best, data


def main2():
    # test fitness of individual

    # Best fitness: 2326
    ind1_str = """
    LIS,FCO,12:18,14:56,172
    MAD,FCO,12:44,14:17,134
    CDG,FCO,11:28,14:40,248
    DUB,FCO,12:34,15:02,109
    BRU,FCO,10:30,14:57,290
    LHR,FCO,12:08,14:59,149
    FCO,LIS,8:04,10:59,136
    FCO,MAD,7:50,10:08,164
    FCO,CDG,8:23,11:07,143
    FCO,DUB,8:23,10:28,149
    FCO,BRU,7:57,11:15,347
    FCO,LHR,8:19,11:16,122"""

    fl1 = [Flight.from_string(1, f.strip()) for f in ind1_str.split('\n')[1:]]
    print(fl1)
    ind1 = Individual(fl1)
    print('IND1: Fitness:', ind1.fitness)
    print(ind1.full_str(False), end='\n\n')
    print(ind1.full_str(True), end='\n\n')

    ind2_str = """
    15,LIS,FCO,16:51,19:09,147
    33,MAD,FCO,15:58,18:40,173
    53,CDG,FCO,15:34,18:11,326
    75,DUB,FCO,17:11,18:30,108
    91,BRU,FCO,13:54,18:02,294
    115,LHR,FCO,17:08,19:08,262
    2,FCO,LIS,8:04,10:59,136
    22,FCO,MAD,7:50,10:08,164
    42,FCO,CDG,8:23,11:07,143
    62,FCO,DUB,8:23,10:28,149
    84,FCO,BRU,9:49,13:51,229
    102,FCO,LHR,8:19,11:16,122"""
    # index, origin, destination, departure, arrival, price
    # remove index from string and add to a list of index
    indexes = [int(i.strip().split(',', 1)[0]) for i in ind2_str.split('\n')[1:]]
    fl2 = [Flight.from_string(indexes[i], ",".join(f.strip().split(',')[1:])) for i, f in
           enumerate(ind2_str.split('\n')[1:])]
    print(fl2)
    ind = Individual(fl2)
    print('IND2: Fitness:', ind.fitness)
    print(ind.full_str(False), end='\n\n')
    print(ind.full_str(True), end='\n\n')


def main():
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
    tests = 30

    # plot parameters
    plot_all = True  # will plot all best individuals of each run
    plot_all_best_fitness = True
    plot_all_fitness_through_generations = True
    plot_all_avg_fitness_through_generations = True

    # Threading parameters
    use_threads = False
    max_threads = 4

    # Data parameters
    data_file = 'flights.txt'
    target_airport = 'FCO'

    # Genetic algorithm parameters
    Individual.PRICE_WEIGHT = 1
    population_size = 1000
    crossover_rate = 0.8
    mutation_rate = 0.05
    genes_to_mutate = None  # None -> all genes can mutate
    generation_size = 50
    tournament_size = 15
    tournament_k = 0.61  # higher k -> higher chance of selecting the best individual in the tournament

    # Read file
    flights = read_file(data_file)

    start = time.time()
    individuals = []  # list of best individuals of each run
    threads = []
    runs_data = []  # list of data of each run (to plot info of each run)
    random.seed()

    if not use_threads:
        for i in range(tests):
            tini = time.time()
            best, data = genetic_algorithm(population_size, generation_size, crossover_rate, mutation_rate,
                                           tournament_size, tournament_k, flights, target_airport, genes_to_mutate)

            print(f'Test({i + 1}): Best fitness: {best.fitness}, takes: {(time.time() - tini):.3f} seconds')
            if plot_all:
                plot_data(data['best'], plot_dir, f'{generation_size} Generations',
                          f'Population size: {population_size},'
                          f' Crossover rate: {crossover_rate},'
                          f' Mutation rate: {mutation_rate}')
            individuals.append(best)
            runs_data.append(data)
    else:
        unique_threads = 0
        while len(individuals) < tests:
            if len(threads) < max_threads and len(individuals) + len(threads) < tests:
                unique_threads += 1
                thread = threading.Thread(target=genetic_algorithm,
                                          args=(population_size, generation_size, crossover_rate,
                                                mutation_rate, tournament_size, tournament_k,
                                                flights, target_airport, genes_to_mutate, runs_data, individuals),
                                          name=f'GA{unique_threads}')
                threads.append(thread)
                thread.start()
            else:
                for thread in threads:
                    time.sleep(1)
                    if not thread.is_alive():  # join finished threads
                        thread.join()
                        print(f'Finished {thread.name}')
                        threads.remove(thread)

    end = time.time()
    runs_results = [individual.fitness for individual in individuals]
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

    individuals_txt = "\n\n".join([f"Individual {i + 1}: {individuals[i].full_str(time_sec=False)}"
                                   for i in range(len(individuals))])
    txt = (f'Date: {date}\n'
           f'Configs:'
           f'\n\ttests={tests}'
           f'\n\tuse_threads={use_threads}'
           f'\n\tmax_threads={max_threads}'
           f'\n\tpopulation_size={population_size}'
           f'\n\tgeneration_size={generation_size}'
           f'\n\tcrossover_rate={crossover_rate}'
           f'\n\tmutation_rate={mutation_rate}'
           f'\n\tgenes_to_mutate={genes_to_mutate}'

           f'\nTournament selection:'
           f'\n\ttournament_size={tournament_size}'
           f'\n\ttournament_k={tournament_k}'

           f'\n\n*{"--*" * 6}  Results *{"--*" * 6}'
           f'\nIt takes {tdelta:.3f} seconds to run {tests} times'
           f'\nAverage time: {avg_time:.3f} seconds'
           f'\nAverage best fitness: {avg}'
           f'\nBest fitness: {best}'
           f'\nWorst fitness: {worst}'
           f'\nStandard deviation: {stdev} (how far from the mean)'
           f'\nVariance: {variance} (square of standard deviation)'
           f'\nMedian: {median} (middle value)'
           f'\n\nBests: {runs_results}'
           f'\n\nIndividuals: {individuals_txt}')
    print(txt)

    # write to file
    if make_reports:
        file_name = f'{date}_best{best}_var{stdev:.2f}_avg{avg:.2f}_ga_p{population_size}' \
                    f'_g{generation_size}_cr{crossover_rate}_mr{mutation_rate}_tz{tournament_size}' \
                    f'_gtm{genes_to_mutate if genes_to_mutate else "N"}_tk{tournament_k:.2f}.txt'
        with open(os.path.join(reports_dir, file_name), 'w') as file:
            file.write(txt)


if __name__ == '__main__':
    main()
