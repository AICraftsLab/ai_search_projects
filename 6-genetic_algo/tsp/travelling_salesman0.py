import os.path
import random
import math
import matplotlib.pyplot as plt
import pickle

from custom_maps import nigeria, africa, world


GENERATIONS = 3000
CITIES = 50
MAP_SIZE = (600, 600)
POPULATION = 500
ELITISM = 10
N_BEST = POPULATION // 3


def distance(city1, city2):
    return math.dist(city1.position, city2.position)


def crossover(genome1, genome2):
    chromosome1 = genome1.chromosome
    chromosome2 = genome2.chromosome

    locus = random.randrange(len(chromosome1))

    new_chromosome1 = chromosome1[:locus] + chromosome2[locus:]
    new_chromosome2 = chromosome2[:locus] + chromosome1[locus:]

    return Genome(new_chromosome1), Genome(new_chromosome2)


def reproduce2(genomes, size):
    offsprings = []

    while len(offsprings) < size:
        parent1, parent2 = random.sample(genomes, 2)
        offspring1, offspring2 = crossover(parent1, parent2)
        offsprings.append(offspring1)
        offsprings.append(offspring2)

    if size % 2 != 0:
        offsprings.pop()

    return offsprings


def reproduce(genomes, size):
    offsprings = []

    while len(offsprings) < size:
        parent = random.choice(genomes)
        offspring = parent.replicate()
        offsprings.append(offspring)

    return offsprings


def plot_solution(genome):
    # Extract city positions and names from the chromosome
    cities = genome.chromosome
    positions = [city.position for city in cities]
    names = [city.name for city in cities]

    # Separate x and y coordinates
    x_coords, y_coords = zip(*positions)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the cities as points
    plt.scatter(x_coords, y_coords, color='blue', s=100, zorder=5)

    # Annotate the cities with their names
    for i, name in enumerate(names):
        plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot the path between cities (i.e., the genome's solution)
    x_coords_with_return = list(x_coords) + [x_coords[0]]
    y_coords_with_return = list(y_coords) + [y_coords[0]]
    plt.plot(x_coords_with_return, y_coords_with_return, color='red', linestyle='-', linewidth=2, zorder=4)

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Traveling Salesman Solution')

    # Show the plot
    plt.show()


def plot_solution_interactive(genome, generation, mutation_prob, save_path=None):
    plt.figure(1)
    # Turn on interactive mode
    if not plt.isinteractive():
        plt.ion()

    # Clear previous plot
    plt.clf()

    # Extract city positions and names from the chromosome
    cities = genome.chromosome
    positions = [city.position for city in cities]
    names = [city.name for city in cities]

    # Separate x and y coordinates
    x_coords, y_coords = zip(*positions)

    # Plot the cities as points
    plt.scatter(x_coords, y_coords, color='blue', s=100, zorder=5)

    # Annotate the cities with their names
    for i, name in enumerate(names):
        plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot the path between cities (i.e., the genome's solution)
    x_coords_with_return = list(x_coords) + [x_coords[0]]
    y_coords_with_return = list(y_coords) + [y_coords[0]]
    plt.plot(x_coords_with_return, y_coords_with_return, color='red', linestyle='-', linewidth=1, zorder=4)

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Traveling Salesman {len(cities)} cities. Generation:{generation} Prob:{mutation_prob} Best:{1/genome.get_fitness():.0f}')

    # Pause to update the plot
    plt.pause(0.1)  # Adjust pause duration to control update speed

    if save_path:
        plt.savefig(save_path)


def plot_overall_distance_interactive(distances, mutation_prob):
    plt.figure(2)
    # Turn on interactive mode
    if not plt.isinteractive():
        plt.ion()

    # Clear previous plot
    plt.clf()

    x_coords = [i for i in range(len(distances))]
    y_coords = distances

    plt.plot(x_coords, y_coords, linestyle='-')

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Overall Distance')
    plt.title(f'Traveling Salesman {len(population.members[0].chromosome)} cities. Generation:{len(distances)} Prob:{mutation_prob}')

    # Pause to update the plot
    plt.pause(0.1)


def plot_compare_overall_distance_interactive(distances_dict, generation, save_path=None):
    plt.figure()

    x_coords = [i for i in range(generation)]

    for name, experiment in distances_dict.items():
        y_coords = experiment
        plt.plot(x_coords, y_coords, label=name, linestyle='-')

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Overall Distance')
    plt.legend()
    plt.title(f'Traveling Salesman Experiments')

    if save_path:
        plt.savefig(save_path)

    # Pause to update the plot
    plt.show()


def stop_interactive_plot():
    plt.ioff()
    plt.show()


class City:
    def __init__(self, x, y, idx, name=None):
        self.position = x, y
        self.idx = idx
        self.name = name if name else f'City{idx}'


class Map:
    def __init__(self, cities, size=MAP_SIZE):
        self.num = cities
        self.size = size
        self.cities = self._generate()

    def _generate(self):
        cities = []
        for i in range(self.num):
            x = random.randrange(self.size[0])
            y = random.randrange(self.size[1])
            city = City(x, y, i)
            cities.append(city)

        return cities

    def save(self, path):
        with open(path, 'w+b') as f:
            pickle.dump(self, f)

    @classmethod
    def from_coordinates_tuples(cls, tuples_list):
        map = Map(0, MAP_SIZE)
        map.num = len(tuples_list)

        for name, longitude, latitude in tuples_list:
            x = latitude
            y = longitude
            city = City(x, y, idx=name, name=name)
            map.cities.append(city)

        return map

    @classmethod
    def load(cls, path):
        with open(path, 'r+b') as f:
            return pickle.load(f)


class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def get_fitness(self):
        fitness = 0
        for i in range(len(self.chromosome)):
            city1 = self.chromosome[i]
            city2 = self.chromosome[(i+1) % len(self.chromosome)]
            fitness += distance(city1, city2)

        return 1 / fitness

    def replicate(self):
        return Genome(self.chromosome.copy())

    def mutate2(self, allele, prob):  # BUG: Lead to division by zero if all genes are same, i.e. same cities
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = random.choice(allele)

    def mutate(self, prob):
        for i in range(len(self.chromosome)):  # TODO: len(self.chromosome) // 2
            if random.random() < prob:
                locus = i

                while locus == i:
                    locus = random.randrange(len(self.chromosome))

                self.chromosome[i], self.chromosome[locus] = self.chromosome[locus], self.chromosome[i]

    def get_solution(self):
        return [city.idx for city in self.chromosome]

    def __repr__(self):
        return f'Distance:{1/self.get_fitness():.0f} Solution:{self.get_solution()}'


class Population:
    def __init__(self, size, map):
        self.size = size
        self.map = map
        self.members = None
        self.best = None

    def generate_members(self, mutation_prob):
        if self.members is None:
            self._initialize()
            return

        best_members = self._select_best_members(N_BEST)
        new_members = reproduce(best_members, self.size - ELITISM)
        for member in new_members:
            # member.mutate2(self.map.cities, MUTATION_PROB)
            member.mutate(mutation_prob)

        new_members.extend(best_members[:ELITISM])

        self.best = best_members[0]
        self.members = new_members

    def _initialize(self):
        self.members = []
        for i in range(self.size):
            genome = Genome(random.sample(self.map.cities, self.map.cities_n))
            self.members.append(genome)

    def _select_best_members(self, n):
        members = [(m, m.get_fitness()) for m in self.members]
        key = lambda x: x[1]
        members = sorted(members, key=key, reverse=True)
        members = members[:n]
        members = [x[0] for x in members]

        return members

    def get_overall_distance(self):
        distance_ = 0

        for member in self.members:
            distance_ += 1 / member.get_fitness()

        return round(distance_, 2)


if __name__ == '__main__':
    experiment_path = 'test_city25'
    os.makedirs(experiment_path, exist_ok=True)
    random.seed(125)

    # map = Map.from_coordinates_tuples(nigeria)
    # map = Map.from_coordinates_tuples(africa)
    # map = Map.from_coordinates_tuples(world)
    map = Map(CITIES)

    best_of_best = (0, None, 0, 0)
    experiment_dict = {}

    for mutation_prob in [0.001, 0.005, 0.008 ,0.01, 0.05, 0.1, 0.15]:
    # for mutation_prob in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        population = Population(POPULATION, map)
        best_fitness, best_fitness_generation = 0, 0
        distances = []

        for i in range(GENERATIONS):
            population.generate_members(mutation_prob)
            best = population.best
            distances.append(population.get_overall_distance())

            if i % 25 == 0 or i == GENERATIONS - 1:
                plot_overall_distance_interactive(distances, mutation_prob)

            if best is not None:
                best_genome_fitness = best.get_fitness()
                if best_genome_fitness > best_fitness:
                    best_fitness = best_genome_fitness
                    best_fitness_generation = i

                if i == 1 or i % 100 == 0 or i == GENERATIONS - 1:
                    plot_solution_interactive(best, generation=i, mutation_prob=mutation_prob)

                if i == GENERATIONS - 1:
                    filename = str(mutation_prob) + '.png'
                    plot_solution_interactive(best, generation=best_fitness_generation, mutation_prob=mutation_prob)
                    plt.savefig(os.path.join(experiment_path, filename))

        experiment_dict[str(mutation_prob)] = distances
        if best_fitness > best_of_best[0]:
            best_of_best = (best_fitness, best, mutation_prob, best_fitness_generation)
        print('Prob:', mutation_prob, 'Best:', 'Generation:', best_fitness_generation, best)
    print()

    print('Overall best:', 'Prob:', best_of_best[2], 'Generation:', best_of_best[3], best_of_best[1])

    best_solution_filename = 'best' + '.png'
    best_solution_filepath = os.path.join(experiment_path, best_solution_filename)
    compare_dist_filename = 'overall_distance' + '.png'
    compare_dist_filepath = os.path.join(experiment_path, compare_dist_filename)

    plot_solution_interactive(best_of_best[1], generation=best_of_best[3], mutation_prob=best_of_best[2],
                              save_path=best_solution_filepath)
    plot_compare_overall_distance_interactive(experiment_dict, GENERATIONS, save_path=compare_dist_filepath)
    stop_interactive_plot()
