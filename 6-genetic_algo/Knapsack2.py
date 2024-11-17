import random

# Problem global variables
GENERATIONS = 500
POPULATION = 100
TOTAL_ITEMS = 30
BAG_CAPACITY = 1000
MUTATION_PROB = 0.01
ELITISM = 10

HIKING_ITEMS = (
    ("Laptop", 95, 150),
    ("Smartphone", 80, 50),
    ("Headphones", 40, 20),
    ("Camera", 60, 120),
    ("Watch", 30, 15),
    ("Tablet", 70, 100),
    ("Charger", 15, 10),
    ("Speaker", 45, 80),
    ("Shoes", 50, 100),
    ("Jacket", 75, 180),
    ("Sunglasses", 25, 30),
    ("Water Bottle", 35, 50),
    ("Tent", 90, 200),
    ("Sleeping Bag", 65, 180),
    ("Camping Stove", 55, 150),
    ("Flashlight", 15, 25),
    ("Map", 10, 10),
    ("Gloves", 20, 15),
    ("Compass", 25, 20),
    ("Binoculars", 60, 100),
    ("Hiking Boots", 85, 160),
    ("Towel", 30, 50),
    ("First Aid Kit", 50, 70),
    ("Power Bank", 35, 60),
    ("Hiking Pole", 40, 100),
    ("Raincoat", 55, 120),
    ("Snacks", 20, 30),
    ("Fire Starter", 40, 20),
    ("Lantern", 55, 40),
    ("Insect Repellent", 20, 15),
    ("Rope", 40, 60),
)


def print_items(items):
    """Func to print bag items info"""
    for item in items:
        print(item)


class Item:
    """Class to represent a single item"""

    def __init__(self, weight, value, name):
        self.weight = weight
        self.value = value
        self.name = name

    def __repr__(self):
        return f'Item {self.name} Value:{self.value} Weight:{self.weight}'

    @classmethod
    def from_random_items(cls, n, min_value=10, max_value=100,
                          min_weight=10, max_weight=BAG_CAPACITY // 5):
        """Creates a random list of items"""
        items = []

        for i in range(n):
            value = random.randrange(min_value, max_value)
            weight = random.randrange(min_weight, max_weight)
            name = str(i)

            item = Item(weight, value, name)
            items.append(item)

        return items

    @classmethod
    def from_items_list(cls, items_list):
        """Creates list of items from (name, value, weight) list"""
        global TOTAL_ITEMS
        items = []
        for name, value, weight in items_list:
            item = Item(weight, value, name)
            items.append(item)

        TOTAL_ITEMS = len(items)
        return items


class Genome:
    ITEMS = None

    def __init__(self, chromosome):
        self.chromosome = chromosome

    def get_fitness(self):
        fitness = 0
        total_weight = 0

        for i in range(len(self.chromosome)):
            gene = self.chromosome[i]
            if gene == 1:
                item = self.ITEMS[i]  # getting item
                fitness += item.value
                total_weight += item.weight

        # punish genomes for exceeding bag capacity
        if total_weight > BAG_CAPACITY:
            fitness = 0

        return fitness

    def get_weight(self):
        total_weight = 0

        for i in range(len(self.chromosome)):
            gene = self.chromosome[i]
            if gene == 1:
                item = self.ITEMS[i]  # getting item
                total_weight += item.weight

        return total_weight

    def mutate(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = int(not self.chromosome[i])  # flips value

    def print_items(self):
        """Prints items selected by this genome"""
        for i, gene in enumerate(self.chromosome):
            if gene == 1:
                item = self.ITEMS[i]
                print(item)

    @classmethod
    def __singlepoint_crossover(cls, genome1_x, genome2_x):
        crossover_point = random.randrange(TOTAL_ITEMS)
        child1_x = genome1_x[:crossover_point] + genome2_x[crossover_point:]
        child2_x = genome2_x[:crossover_point] + genome1_x[crossover_point:]

        return Genome(child1_x), Genome(child2_x)

    @classmethod
    def __doublepoint_crossover(cls, genome1_x, genome2_x):
        points = random.sample(range(TOTAL_ITEMS), k=2)
        points.sort()
        point1, point2 = points
        child1_x = genome1_x[:point1] + genome2_x[point1:point2] + genome1_x[point2:]
        child2_x = genome2_x[:point1] + genome1_x[point1:point2] + genome2_x[point2:]

        return Genome(child1_x), Genome(child2_x)

    @classmethod
    def crossover(cls, genome1_x, genome2_x, c_type='single'):
        if c_type == 'single':
            return cls.__singlepoint_crossover(genome1_x, genome2_x)
        elif c_type == 'double':
            return cls.__doublepoint_crossover(genome1_x, genome2_x)


class Population:
    """Class to represent a population"""

    def __init__(self, size):
        self.size = size
        self.genomes = []
        self.__initialize()

    def __initialize(self):
        """Initialize population with random genomes"""
        for _ in range(self.size):
            chromosome = [random.randrange(2) for _ in range(TOTAL_ITEMS)]
            genome = Genome(chromosome)
            self.genomes.append(genome)

    def generate_next_generation(self):
        """Generates the next generation by evolution"""
        next_gen_genomes = []

        parents_n = POPULATION - ELITISM
        parents = self.__select_parents(parents_n)  # selection

        for i in range(0, parents_n, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            # crossover & mutation
            child1, child2 = Genome.crossover(parent1.chromosome, parent2.chromosome)
            child1.mutate(MUTATION_PROB)
            child2.mutate(MUTATION_PROB)
            next_gen_genomes.append(child1)
            next_gen_genomes.append(child2)

        elites = self.__get_elites(ELITISM)  # elitism
        next_gen_genomes.extend(elites)
        self.genomes = next_gen_genomes

        return elites[0]  # return best genome, top elite

    def __get_elites(self, elitism):
        key = lambda x: x.get_fitness()
        genomes = sorted(self.genomes, key=key, reverse=True)
        genomes = genomes[:elitism]

        return genomes

    def __select_parents(self, n):
        key = lambda x: x.get_fitness()
        genomes = sorted(self.genomes, key=key, reverse=True)
        genomes = genomes[:POPULATION // 4]

        parents = random.choices(genomes, k=n)

        return parents


if __name__ == '__main__':
    # items = Item.from_random_items(TOTAL_ITEMS)
    items = Item.from_items_list(HIKING_ITEMS)
    Genome.ITEMS = items
    population = Population(POPULATION)

    # keeping track of best
    overall_best = None
    overall_best_fitness = 0
    overall_best_weight = 0

    for i in range(GENERATIONS):
        best = population.generate_next_generation()
        best_fitness = best.get_fitness()

        # Check for new best
        if overall_best is None or best_fitness > overall_best_fitness:
            overall_best = best
            overall_best_fitness = best_fitness
            overall_best_weight = best.get_weight()

        print('Generation:', i, 'Best Fitness:', overall_best_fitness, 'Weight:', overall_best_weight)
    print()

    print('All items')
    print_items(items)
    print()

    print('Overall Best Fitness:', overall_best_fitness, overall_best.chromosome)
    overall_best.print_items()
    print('Total weight', overall_best.get_weight())
