import pandas as pd


def hour_to_min(hour: str):
    return int(hour.split(':')[0]) * 60 + int(hour.split(':')[1])


def min_to_hour(minutes: int):
    return str(minutes // 60) + ':' + str(minutes % 60)


class Flight:
    def __init__(self, index: int, dep_airport: str, arr_airport: str, dep_time: int, arr_time: int, price: int):
        self.id = index
        self.dep_airport = dep_airport
        self.arr_airport = arr_airport
        self.dep_time = dep_time
        self.arr_time = arr_time
        self.price = price

    def dep_time_show(self):
        return min_to_hour(self.dep_time)

    def arr_time_show(self):
        return min_to_hour(self.arr_time)

    def __str__(self):
        return f'Flight({self.id}): ' \
               f'{self.dep_airport} {self.arr_airport} {self.dep_time_show()} {self.arr_time_show()} {self.price}'

    def __repr__(self):
        return str(self)


class Organism:
    def __init__(self, flights: list):
        self.flights = flights
        self.fitness = 0

    def __str__(self):
        return f'Organism: {self.fitness}'

    def __repr__(self):
        return str(self)


def read_file(path):
    flights = []
    df = pd.read_csv(path, sep=',', header=None)
    for index, row in enumerate(df.iterrows()):
        # FCO,LIS,6:19,8:13,239
        flights.append(Flight(index, row[1][0], row[1][1], hour_to_min(row[1][2]), hour_to_min(row[1][3]), row[1][4]))
    return flights


def main():
    flights = read_file('flights.txt')
    # TODO continue, create logic of first population, index jump to 20 to change


if __name__ == '__main__':
    main()
