import unittest

from main import Flight, Individual


class GeneticAlgorithm(unittest.TestCase):
    def test_best_fitness(self):
        file = 'best_ind.txt'
        expected_fitness = 2326

        flights = []
        with open(file, 'r') as f:
            for index, flight in enumerate(f.read().strip().split('\n')):
                flights.append(Flight.from_string(index, flight.strip()))

        ind1 = Individual(flights, itype='Complete')
        print('IND1: Fitness:', ind1.fitness)
        print('Idividual: time in hours')
        print(ind1.full_str(False), end='\n\n')
        print('Idividual: time in minutes')
        print(ind1.full_str(True), end='\n\n')
        self.assertEqual(ind1.fitness, expected_fitness)


if __name__ == '__main__':
    unittest.main()
