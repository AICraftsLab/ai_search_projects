import os.path
import random
import math
import matplotlib.pyplot as plt
import pickle

from custom_maps import nigeria, africa, world


GENERATIONS = 100
CITIES = 40
MAP_SIZE = (300, 300)
POPULATION = 200
ELITISM = 10

TOP_K_RANDOM_SELECTION = POPULATION // 4
TOURNAMENT_SIZE = POPULATION // 10
SCRAMBLE_SIZE = int(CITIES * 0.2)


def distance(city1, city2):
    return math.dist(city1.position, city2.position)


def plot_solution_interactive(genome, generation, title='', save_path=None):
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
    plt.plot(x_coords_with_return, y_coords_with_return, color='red', linestyle='-', zorder=4)

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'TSP: {len(cities)} cities. {title} Generation:{generation} Best:{1/genome.get_fitness():.2f}')

    # Pause to update the plot
    plt.pause(0.1)  # Adjust pause duration to control update speed

    if save_path:
        plt.savefig(save_path)


def plot_overall_distance_interactive(distances, title=''):
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
    plt.title(f'TSP: {len(population.members[0].chromosome)} cities. {title} Generation:{len(distances)}')

    # Pause to update the plot
    plt.pause(0.1)


def plot_compare_overall_distance(experiment_dict, generations, save_path=None):
    plt.figure()

    x_coords = [i for i in range(generations)]

    for name, data in experiment_dict.items():
        y_coords = data['distances']
        plt.plot(x_coords, y_coords, label=name, linestyle='-')

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Overall Distance')
    plt.legend()
    plt.title(f'TSP Overall Distances.')

    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_compare_best_history(experiment_dict, generations, save_path=None):
    plt.figure()

    for name, data in experiment_dict.items():
        y_coords, x_coords = zip(*data['best_history'])
        
        x_coords = list(x_coords)
        x_coords.append(generations)
        
        y_coords = list(y_coords)
        y_coords.append(y_coords[-1])
        
        plt.plot(x_coords, y_coords, label=name, linestyle='-')

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.legend()
    plt.title(f'TSP Best Solution History')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def stop_interactive_plot():
    plt.ioff()
    plt.show()


def print_experiment_summary(exp_dict, save_path):
    exp_list = []
    if save_path:
        summary_file = open(save_path, 'w')
        
    for title, data in exp_dict.items():
        exp_list.append((title, data['best'], data['best_fitness'], data['best_fitness_gen']))

    key = lambda x: x[2]
    exp_list = sorted(exp_list, key=key, reverse=True)

    print('Best Solutions')
    for i, exp in enumerate(exp_list, start=1):
        print(i, exp[3], exp[1], exp[0])
        if save_path:
            print(i, exp[3], exp[1], exp[0], file=summary_file)
    summary_file.close()


class City:
    def __init__(self, x, y, idx, name=None):
        self.position = x, y
        self.idx = idx
        self.name = name if name else f'City{idx}'

    def __repr__(self):
        return self.idx


class Map:
    def __init__(self, cities, size=MAP_SIZE):
        self.cities_n = cities
        self.size = size
        self.cities = self._generate()

    def _generate(self):
        cities = []
        for i in range(self.cities_n):
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
        map.cities_n = len(tuples_list)

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

    def mutate(self, type):
        if type == 'swap':
            self._swap_mutation()
        elif type == 'inverse':
            self._inversion_mutation()
        elif type == '2-opt':
            self._two_opt_mutation()
        elif type == 'random':
            rand = random.randrange(4)
            if rand == 0:
                self._swap_mutation()
            elif rand == 1:
                self._inversion_mutation()
            elif rand == 2:
                self._two_opt_mutation()
        else:
            raise Exception('Invalid mutation type')

    def _swap_mutation(self):
        alleles = random.sample(range(len(self.chromosome)), k=2)
        a1, a2 = alleles
        self.chromosome[a1], self.chromosome[a2] = self.chromosome[a2], self.chromosome[a1]

    def _inversion_mutation(self):
        alleles = random.sample(range(len(self.chromosome)), k=2)
        alleles.sort()
        a1, a2 = alleles
        genes = self.chromosome[a1:a2]
        genes.reverse()
        new_chromosome = self.chromosome[:a1] + genes + self.chromosome[a2:]
        self.chromosome = new_chromosome

    def _two_opt_mutation(self):
        alleles = random.sample(range(len(self.chromosome)), k=2)
        a1, a2 = alleles
        edge1 = a1, (a1 + 1) % len(self.chromosome)
        edge2 = a2, (a2 + 1) % len(self.chromosome)
        self.chromosome[edge1[1]], self.chromosome[edge2[0]] = self.chromosome[edge2[0]], self.chromosome[edge1[1]]

    def get_solution(self):
        return [city.idx for city in self.chromosome]

    def __repr__(self):
        return f'Distance:{1/self.get_fitness():.2f} Solution:{self.get_solution()}'

    @classmethod
    def order_crossover(cls, parent1, parent2):
        pop = [x for x in range(len(parent1.chromosome))]
        loci = random.sample(pop, k=2)
        loci.sort()
        l1, l2 = loci

        child1_chromosome = [None] * len(parent1.chromosome)
        child2_chromosome = [None] * len(parent2.chromosome)
        child1_chromosome[l1:l2] = parent2.chromosome[l1:l2]
        child2_chromosome[l1:l2] = parent1.chromosome[l1:l2]

        def orderly_insert(parent_x, child_x, locus2):
            swapped_parent_x = []
            for i in range(locus2, locus2 + len(parent_x)):
                swapped_parent_x.append(parent_x[i % len(parent_x)])

            l = locus2
            for gene in swapped_parent_x:
                if gene not in child_x:
                    child_x[l % len(child_x)] = gene
                    l += 1

        orderly_insert(parent1.chromosome, child1_chromosome, l2)
        orderly_insert(parent2.chromosome, child2_chromosome, l2)

        return Genome(child1_chromosome), Genome(child2_chromosome)


class Population:
    def __init__(self, size, map):
        self.size = size
        self.map = map
        self.members = None
        self.best = None

    def generate_members(self, s_type='roulette', m_type='swap'):
        if self.members is None:
            self._initialize()
            return

        parents = self._select_parents(self.size - ELITISM, type=s_type)
        new_members = []
        for i in range(self.size - ELITISM):
            parent = parents[i]
            offspring = parent.replicate()
            # offspring = Genome.order_crossover(parents[i], parents[i+1])
            offspring.mutate(type=m_type)
            new_members.append(offspring)

        sorted_members = self._sort_members()
        new_members.extend(sorted_members[-ELITISM:])

        self.best = sorted_members[-1]
        self.members = new_members

    def _initialize(self):
        self.members = []
        for i in range(self.size):
            genome = Genome(random.sample(self.map.cities, self.map.cities_n))
            self.members.append(genome)

    def _sort_members(self):
        members = [(m, m.get_fitness()) for m in self.members]
        key = lambda x: x[1]
        members = sorted(members, key=key)
        members = [x[0] for x in members]

        return members

    def _top_k_random_selection(self, n, k):
        members = self._sort_members()
        top_k = members[-k:]
        parents = random.choices(top_k, k=n)

        return parents

    def _roulette_selection(self, n, **kwargs):
        members_fitness = [m.get_fitness() for m in self.members]
        fitness_sum = sum(members_fitness)
        probs = [x / fitness_sum for x in members_fitness]
        parents = random.choices(self.members, weights=probs, k=n)

        return parents

    def _rank_selection(self, n, **kwargs):
        members = self._sort_members()

        size = len(members)
        ranks_sum = int((size + 1) * (size / 2))
        probs = [x / ranks_sum for x in range(1, size + 1)]
        parents = random.choices(members, weights=probs, k=n)

        return parents

    def _tournament_selection(self, n, t_size):
        parents = []

        def tournament(participants):
            winner = (participants[0], participants[0].get_fitness())

            for p in participants[1:]:
                p_fitness = p.get_fitness()
                if p_fitness > winner[1]:
                    winner = (p, p_fitness)

            return winner[0]

        while len(parents) < n:
            participants = random.choices(self.members, k=t_size)
            winner = tournament(participants)
            parents.append(winner)

        return parents

    def _select_parents(self, n, type):
        if type == 'top_k':
            return self._top_k_random_selection(n, k=TOP_K_RANDOM_SELECTION)
        elif type == 'roulette':
            return self._roulette_selection(n)
        elif type == 'rank':
            return self._rank_selection(n)
        elif type == 'tournament':
            return self._tournament_selection(n, t_size=TOURNAMENT_SIZE)
        else:
            raise Exception('Invalid selection type')


if __name__ == '__main__':
    seed = 12.7
    experiment_path = 'test'
    os.makedirs(experiment_path, exist_ok=True)
    random.seed(seed)

    # map = Map.from_coordinates_tuples(nigeria)
    # map = Map.from_coordinates_tuples(africa)
    # map = Map.from_coordinates_tuples(world)
    map = Map(CITIES)

    top_best = None
    top_best_fitness = 0
    experiment_dict = {}

    for s_type in ['top_k', 'roulette', 'rank', 'tournament']:
        for m_type in ['swap', 'inverse', '2-opt', 'random']:
            random.seed(seed)
            title = s_type + ' ' + m_type
            population = Population(POPULATION, map)
            best_fitness, best_fitness_generation = 0, 0
            best_history = []

            for i in range(GENERATIONS):
                population.generate_members(s_type=s_type, m_type=m_type)
                best = population.best

                if best is not None:
                    best_genome_fitness = best.get_fitness()
                    if best_genome_fitness > best_fitness:
                        best_fitness = best_genome_fitness
                        best_fitness_generation = i
                        best_history.append((round(1 / best_fitness, 2), best_fitness_generation))

                    if i == 1 or i % 100 == 0 or i == GENERATIONS - 1:
                        plot_solution_interactive(best, generation=i, title=title)

                    if i == GENERATIONS - 1:
                        filename = title + '.png'
                        plot_solution_interactive(best, generation=best_fitness_generation, title=title)
                        plt.savefig(os.path.join(experiment_path, filename))

            experiment_dict[title] = dict(best_history=best_history, best=best, best_fitness=best_fitness,
                                          best_fitness_gen=best_fitness_generation)

            if best_fitness > top_best_fitness:
                top_best_fitness = best_fitness
                top_best = best
            print(title, 'Generation:', best_fitness_generation, best)
        print()
    
    best_solution_filepath = os.path.join(experiment_path, 'best.png')
    compare_best_filepath = os.path.join(experiment_path, 'best_solutions.png')
    experiment_summary_path = os.path.join(experiment_path, 'summary.txt')

    print_experiment_summary(experiment_dict, save_path=experiment_summary_path)
    plot_solution_interactive(top_best, generation=-1, save_path=best_solution_filepath)
    plot_compare_best_history(experiment_dict, GENERATIONS, save_path=compare_best_filepath)
    stop_interactive_plot()
