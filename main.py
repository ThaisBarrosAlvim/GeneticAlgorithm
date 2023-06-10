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

        fitness = 0

        # calculate the time gap between the flights that arrival in the target airport
        greatest_arr_time = self.flights[0].arr_time
        lowest_arr_time = self.flights[0].arr_time
        fitness += self.flights[0].price * self.PRICE_WEIGHT
        for i in self.flights[1:6]:  # type: Flight
            fitness += i.price * self.PRICE_WEIGHT
            if i.arr_time > greatest_arr_time:
                greatest_arr_time = i.arr_time
            if i.arr_time < lowest_arr_time:
                lowest_arr_time = i.arr_time
        fitness += greatest_arr_time - lowest_arr_time

        # calculate the time gap between the flights that depart from the target airport
        greatest_dep_time = self.flights[6].dep_time
        lowest_dep_time = self.flights[6].dep_time
        fitness += self.flights[6].price * self.PRICE_WEIGHT
        for i in self.flights[7:]:  # type: Flight
            fitness += i.price * self.PRICE_WEIGHT
            if i.dep_time > greatest_dep_time:
                greatest_dep_time = i.dep_time
            if i.dep_time < lowest_dep_time:
                lowest_dep_time = i.dep_time
        fitness += greatest_dep_time - lowest_dep_time

        return fitness

    def __repr__(self):
        return str(self)


def read_file(path):
    flights = defaultdict(list)
    df = pd.read_csv(path, sep=',', header=None)
    for index, row in enumerate(df.iterrows()):
        # FCO,LIS,6:19,8:13,239
        flights[f'{row[1][0]}|{row[1][1]}'].append(Flight(index, row[1][0], row[1][1], hour_to_min(row[1][2]),
                                                          hour_to_min(row[1][3]), row[1][4]))
    return flights


def create_individual(flights: dict, target_airport: str):
    organism = Individual()
    random.seed()
    # add random flights in order of arrive in TARGET_AIRPORT and then depart from TARGET_AIRPORT
    for k in sorted(flights, key=lambda x: target_airport == x[4:], reverse=True):
        organism.insert_flight(random.choice(flights[k]))

    return organism


def tournament_selection(population: list, size: int, k: float = 0.75):
    new_population = set()
    for i in range(len(population)):
        tournament = random.sample(population, size)
        # the goal is to minimize the fitness
        if random.random() < k:
            winner = min(tournament, key=lambda x: x.fitness)
        else:
            winner = max(tournament, key=lambda x: x.fitness)
        new_population.add(winner)
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
    for i in range(len(population)):
        if random.random() < crossover_rate and len(population) > 1:
            parent1, parent2 = random.sample(population, 2)
            population.remove(parent1)
            population.remove(parent2)
            child1, child2 = uniform_crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)
        elif len(population) >= 1:
            new_population.append(population[0])
    return new_population


def bit_flip_mutation(organism: Individual, flights: dict, flights_to_mutate: int = 3):
    for i in range(flights_to_mutate):
        selected_to_mutate = random.randint(0, len(organism.flights) - 1)
        new_flight = random.choice(flights[organism.flights[selected_to_mutate].get_group()])
        organism.insert_flight(new_flight, selected_to_mutate)
    return organism


def plot_data(data: list, context_title: str = '', context_data: str = '', multiple_plots=False,
              ylabel='Fitness', xlabel='Generation') -> None:
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
    plt.show()


def genetic_algorithm(population_size: int, generation_size: int, crossover_rate: float, mutation_rate: float,
                      tournament_size: int, tournament_k: float, flights_to_mutate: int, flights: dict,
                      target_airport: str, run_data: list = None, individuals: list = None):
    population = []

    for i in range(population_size):
        population.append(create_individual(flights, target_airport))

    data = {'avg_fitness': [], 'best': [], 'worst': []}
    for i in range(generation_size):
        population_to_crossover = tournament_selection(population, tournament_size, tournament_k)

        best = min(population, key=lambda x: x.fitness)
        data['avg_fitness'].append(sum([x.fitness for x in population]) / len(population))
        data['best'].append(best.fitness)
        data['worst'].append(max(population, key=lambda x: x.fitness).fitness)
        population.remove(best)

        population_crossover = crossover_population(list(population_to_crossover), crossover_rate)
        population_remaining = list(set(population) - set(population_to_crossover))

        mutated_population = []
        for individual in population_remaining:
            if random.random() < mutation_rate:
                population_remaining.remove(individual)
                mutated_population.append(bit_flip_mutation(individual, flights, flights_to_mutate))

        population = list(population_crossover) + mutated_population + population_remaining
        population.append(best)

    best = min(population, key=lambda x: x.fitness)
    if individuals is not None and run_data is not None:
        individuals.append(best)
        run_data.append(data)
    else:
        return best, data


def main2():
    # test fitness of individual
    # 10 LIS,FCO,12:18,14:56,172
    # 30 MAD,FCO,12:44,14:17,134
    # 48 CDG,FCO,11:28,14:40,248
    # 70 DUB,FCO,12:34,15:02,109
    # 88 BRU,FCO,10:30,14:57,290
    # 110 LHR,FCO,12:08,14:59,149
    # 7 FCO,LIS,11:07,13:24,171
    # 27 FCO,MAD,10:33,13:11,132
    # 47 FCO,CDG,11:08,14:38,262
    # 67 FCO,DUB,10:33,12:03,74
    # 87 FCO,BRU,10:51,14:16,256
    # 107 FCO,LHR,10:32,13:16,139
    fl = [Flight(1, 'LIS', 'FCO', hour_to_min('12:18'), hour_to_min('14:56'), 172),
          Flight(1, 'MAD', 'FCO', hour_to_min('12:44'), hour_to_min('14:17'), 134),
          Flight(1, 'CDG', 'FCO', hour_to_min('11:28'), hour_to_min('14:40'), 248),
          Flight(1, 'DUB', 'FCO', hour_to_min('12:34'), hour_to_min('15:02'), 109),
          Flight(1, 'BRU', 'FCO', hour_to_min('10:30'), hour_to_min('14:57'), 290),
          Flight(1, 'LHR', 'FCO', hour_to_min('12:08'), hour_to_min('14:59'), 149),
          Flight(1, 'FCO', 'LIS', hour_to_min('11:07'), hour_to_min('13:24'), 171),
          Flight(1, 'FCO', 'MAD', hour_to_min('10:33'), hour_to_min('13:11'), 132),
          Flight(1, 'FCO', 'CDG', hour_to_min('11:08'), hour_to_min('14:38'), 262),
          Flight(1, 'FCO', 'DUB', hour_to_min('10:33'), hour_to_min('12:03'), 74),
          Flight(1, 'FCO', 'BRU', hour_to_min('10:51'), hour_to_min('14:16'), 256),
          Flight(1, 'FCO', 'LHR', hour_to_min('10:32'), hour_to_min('13:16'), 139)
          ]
    print(fl)
    ind = Individual(fl)
    print(ind.full_str(False), end='\n\n')
    print(ind.full_str(True))
    print(ind.fitness)


def main():
    # Test parameters
    reports_dir = 'reports'
    make_reports = True  # create file with all test results or print
    tests = 30
    plot_all = False  # will plot all best individuals of each run
    use_threads = False
    max_threads = 4

    # Data parameters
    data_file = 'flights.txt'
    target_airport = 'FCO'

    # Genetic algorithm parameters
    Individual.PRICE_FACTOR = 1
    population_size = 20
    crossover_rate = 0.7
    mutation_rate = 0.05
    flights_to_mutate = 1
    generation_size = 350
    tournament_size = 6
    tournament_k = 0.99  # higher k -> higher chance of selecting the best individual in the tournament

    # Read file
    flights = read_file(data_file)

    start = time.time()
    individuals = []  # list of best individuals of each run
    threads = []
    runs_data = []  # list of data of each run (to plot info of each run)
    if not use_threads:
        for i in range(tests):
            tini = time.time()
            best, data = genetic_algorithm(population_size, generation_size, crossover_rate, mutation_rate,
                                           tournament_size, tournament_k, flights_to_mutate, flights, target_airport)

            print(f'Test({i + 1}): Best fitness: {best.fitness}, takes: {(time.time() - tini):.3f} seconds')
            individuals.append(best)
            runs_data.append(data)
    else:
        unique_threads = 0
        while len(individuals) < tests:
            if len(threads) < max_threads and len(individuals) + len(threads) < tests:
                unique_threads += 1
                thread = threading.Thread(target=genetic_algorithm,
                                          args=(population_size, generation_size, crossover_rate,
                                                mutation_rate, tournament_size, tournament_k, flights_to_mutate,
                                                flights, target_airport, runs_data, individuals),
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
    # plot data of each run
    if plot_all:
        for run_data in runs_data:
            plot_data(run_data['best'], f'{generation_size} Generations', f'Population size: {population_size},'
                                                                          f' Crossover rate: {crossover_rate},'
                                                                          f' Mutation rate: {mutation_rate}')
    # plot data of all runs
    plot_data(runs_results, f'All Tests Best Fitness', f'Population size: {population_size},'
                                                       f' Crossover rate: {crossover_rate},'
                                                       f' Mutation rate: {mutation_rate}',
              ylabel='Fitness', xlabel='Test number')
    # plot all in one graph
    plot_data([rd['best'] for rd in runs_data],
              f'All Tests Best Fitness', f'Population size: {population_size},'
                                         f' Crossover rate: {crossover_rate},'
                                         f' Mutation rate: {mutation_rate}', multiple_plots=True)

    plot_data([rd['avg_fitness'] for rd in runs_data], f'All Tests Average Fitness',
              f'Population size: {population_size},'
              f' Crossover rate: {crossover_rate},'
              f' Mutation rate: {mutation_rate}', ylabel='Average Fitness', xlabel='Generation', multiple_plots=True)

    # Write results to file
    avg = sum(runs_results) / len(runs_results)
    best = min(runs_results)
    worst = max(runs_results)
    stdev = statistics.stdev(runs_results)
    variance = statistics.variance(runs_results)
    median = statistics.median(runs_results)
    tdelta = end - start
    avg_time = tdelta / tests
    date = datetime.datetime.today().strftime("%d-%m_%H:%M")

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
           f'\n\tflights_to_mutate={flights_to_mutate}'

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
        # check if dir exists
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        file_name = f'{reports_dir}/{date}_best{best}_var{stdev:.2f}_avg{avg:.2f}_ga_p{population_size}' \
                    f'_g{generation_size}_cr{crossover_rate}_mr{mutation_rate}.txt'
        with open(file_name, 'w') as file:
            file.write(txt)


if __name__ == '__main__':
    main()
