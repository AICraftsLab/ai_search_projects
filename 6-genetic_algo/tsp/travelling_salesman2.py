import os.path
import random
import math
import matplotlib.pyplot as plt
import pickle

from custom_maps import nigeria, africa, world

# Version: Crossover Version

# Problem global variables 
GENERATIONS = 1000
CITIES = 30
MAP_SIZE = (300, 300)
POPULATION = 100
ELITISM = 10
TOP_K = POPULATION // 4


def plot_solution_interactive(genome, generation, title='', save_path=None):
    # Get figure number 1 or create it if not exist
    plt.figure(1)

    if not plt.isinteractive():
        plt.ion()

    plt.clf()

    # Extract city positions and names from the chromosome
    cities = genome.chromosome
    positions = [city.position for city in cities]
    names = [city.name for city in cities]

    # Separate x and y coordinates
    x_coords, y_coords = zip(*positions)

    # Plot the cities
    plt.scatter(x_coords, y_coords, color='blue', s=100, zorder=1)

    # Annotate the cities with their names
    for i, name in enumerate(names):
        plt.annotate(name, (x_coords[i], y_coords[i]),
                     textcoords="offset points",
                     xytext=(0, 10), ha='center')

    # Plot the path between cities
    x_coords = list(x_coords) + [x_coords[0]]
    y_coords = list(y_coords) + [y_coords[0]]
    plt.plot(x_coords, y_coords , color='red', zorder=0)

    plt.axis('off')
    plt.title(f'TSP: {len(cities)} cities. {title} Generation:{generation} '
              f'Best:{1/genome.get_fitness():.2f}')

    # Pause to update the plot
    plt.pause(0.01)
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)


def plot_compare_best_history(experiment_dict, save_path=None):
    """Func to compare best genomes gotten overtime"""
    plt.figure()

    # Plotting each run info
    for name, data in experiment_dict.items():
        y_coords, x_coords = zip(*data['best_history'])

        # Append GENERATIONS to extend the plot line
        x_coords = list(x_coords)
        x_coords.append(GENERATIONS)

        # Append last best solution to extend the plot line
        y_coords = list(y_coords)
        y_coords.append(y_coords[-1])
        
        plt.plot(x_coords, y_coords, label=name)

    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.title('TSP Best Solutions History')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def stop_interactive_plot():
    """Turn off interactive mode"""
    plt.ioff()
    plt.show()  # redraw last plot


def distance(city1, city2):
    """Calculate the distance between two cities"""
    return math.dist(city1.position, city2.position)


def print_experiment_summary(exp_dict, save_path=None):
    """Print (and save) experiment result summary"""
    
    if save_path:
        summary_file = open(save_path, 'w')
    
    runs_list = []
    
    # Create a (title, best, best_fitness, best_fitness_generation)
    # list for each run
    for run_title, run_data in exp_dict.items():
        runs_list.append((run_title, run_data['best'],run_data['best_fitness'],
                          run_data['best_fitness_gen']))

    key = lambda x: x[2]  # Sort by fitness (descending)
    runs_list = sorted(runs_list, key=key, reverse=True)

    print('Best Solutions')
    for i, run in enumerate(runs_list, start=1):
        # Write to stdout/console
        # Rank, Generation it emerged, distance and chromosome, run title
        print(i, 'Gen:', run[3], run[1], run[0])
        
        # If save_path is specified, write to the file
        if save_path:
            print(i, 'Gen:', run[3], run[1], run[0], file=summary_file)

    if save_path:
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
        self.cities = self.__generate()

    def __generate(self):
        """Generate a random map"""
        cities = []
        for i in range(self.cities_n):
            x = random.randrange(self.size[0])
            y = random.randrange(self.size[1])
            city = City(x, y, i)
            cities.append(city)

        return cities

    def save(self, path):
        """Save map to a file"""
        with open(path, 'w+b') as f:
            pickle.dump(self, f)

    @classmethod
    def from_coordinates_tuples(cls, tuples_list):
        """Create map from (name, longitude, latitude) list"""
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
        """Load a map file"""
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
        """Create a replica of this genome"""
        return Genome(self.chromosome.copy())

    def mutate(self):
        points = random.sample(range(len(self.chromosome)), k=2)
        p1, p2 = points 
        self.chromosome[p1], self.chromosome[p2] = self.chromosome[p2], self.chromosome[p1]

    def get_solution(self):
        """Return a list of city indices in chromosome"""
        return [city.idx for city in self.chromosome]

    def __repr__(self):
        """String representation of genome"""
        return f'Distance:{1/self.get_fitness():.2f} Solution:{self.get_solution()}'

    @classmethod
    def __order_crossover(cls, parent1, parent2):
        """Order Crossover, ox"""
        points = random.sample(range(len(parent1.chromosome)), k=2)
        points.sort()
        p1, p2 = points 

        child1_chromosome = [None] * len(parent1.chromosome)
        child2_chromosome = [None] * len(parent2.chromosome)
        child1_chromosome[p1:p2] = parent2.chromosome[p1:p2]
        child2_chromosome[p1:p2] = parent1.chromosome[p1:p2]

        def orderly_insert(parent_x, child_x):
            """Orderly insert parent_x into child_x"""
            reordered_parent_x = []  # parent new list
            for i in range(p2, p2 + len(parent_x)):  # start at p2
                reordered_parent_x.append(parent_x[i % len(parent_x)])

            # Fill child_x with parent_x starting from p2
            p = p2
            for gene in reordered_parent_x:
                if gene not in child_x:
                    child_x[p % len(child_x)] = gene
                    p += 1

        orderly_insert(parent1.chromosome, child1_chromosome)
        orderly_insert(parent2.chromosome, child2_chromosome)

        return Genome(child1_chromosome), Genome(child2_chromosome)

    @classmethod
    def __partially_mapped_crossover(cls, parent1, parent2):
        """Partially-Mapped Crossover, pmx"""
        points = random.sample(range(len(parent1.chromosome)), k=2)
        points.sort()
        p1, p2 = points

        parent1_segment = parent1.chromosome[p1:p2]
        parent2_segment = parent2.chromosome[p1:p2]

        # Create mappings (composite mapping not addressed here)
        mappings = list(zip(parent1_segment, parent2_segment))

        child1_x = [None] * len(parent1.chromosome)
        child2_x = [None] * len(parent1.chromosome)
        child1_x[p1:p2] = parent2_segment
        child2_x[p1:p2] = parent1_segment

        def get_gene_counterpart(gene, mappings):
            """Checks for a gene mapping. Recursively to address composite mapping"""
            for mapped_values in mappings:
                if gene in mapped_values:
                    gene = mapped_values[0] if mapped_values[0] is not gene \
                        else mapped_values[1]
                    mappings.remove(mapped_values)
                    return get_gene_counterpart(gene, mappings)  # Address composite mapping
            return gene

        def fill_chromosome(chromosome, parent_x):
            """Fill chromosome with parent's chromosome"""
            for i in range(len(parent_x)):
                if i < p1 or i >= p2:  # Ignore genes btw p1 and p2
                    gene = parent_x[i]
                    if gene not in chromosome:
                        chromosome[i] = gene
                    else:
                        chromosome[i] = get_gene_counterpart(gene, mappings.copy())

        fill_chromosome(child1_x, parent1.chromosome)
        fill_chromosome(child2_x, parent2.chromosome)

        return Genome(child1_x), Genome(child2_x)

    @classmethod
    def __cycle_crossover(cls, parent1, parent2):
        """Cycle Crossover, cx"""
        index = 0
        child1_x = [None] * len(parent1.chromosome)
        child2_x = [None] * len(parent1.chromosome)
        parent1_first_gene = parent1.chromosome[index]
        parent1_gene = parent1_first_gene
        cycled = False

        while not cycled:
            # Get gene at index in parent 2
            parent2_gene = parent2.chromosome[index]

            # Fill offspring positions
            child1_x[index] = parent1_gene
            child2_x[index] = parent2_gene

            # Updates
            index = parent2.chromosome.index(parent1_gene)
            parent1_gene = parent1.chromosome[index]
            cycled = parent1_gene == parent1_first_gene

        def fill_chromosome(chromosome, parent_x):
            """Fill chromosome with parent's chromosome"""
            for i in range(len(chromosome)):
                gene = chromosome[i]
                if gene is None:  # Fill only empty positions
                    chromosome[i] = parent_x[i]

        fill_chromosome(child1_x, parent2.chromosome)
        fill_chromosome(child2_x, parent1.chromosome)

        return Genome(child1_x), Genome(child2_x)
    
    @classmethod
    def crossover(cls, parent1, parent2, c_type):
        """Perform crossover btw 2 parents"""
        if c_type == 'ox':
            return cls.__order_crossover(parent1, parent2)
        elif c_type == 'pmx':
            return cls.__partially_mapped_crossover(parent1, parent2)
        elif c_type == 'cx':
            return cls.__cycle_crossover(parent1, parent2)
            
            
class Population:
    def __init__(self, size, map):
        self.size = size
        self.map = map
        self.members = []
        self.__initialize()

    def __initialize(self):
        for i in range(self.size):
            genome = Genome(random.sample(self.map.cities, self.map.cities_n))
            self.members.append(genome)

    def generate_next_generation(self, c_type='ox'):
        next_gen_members = []
        parents_n = self.size - ELITISM
        parents = self.__top_k_selection(parents_n, TOP_K)

        for i in range(0, parents_n, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover(parent1, parent2, c_type=c_type)
            child1.mutate()
            child2.mutate()
            next_gen_members.append(child1)
            next_gen_members.append(child2)

        sorted_members = self.__sort_members()
        next_gen_members.extend(sorted_members[-ELITISM:])

        self.members = next_gen_members

        return sorted_members[-1]

    def __sort_members(self):
        key = lambda x: x.get_fitness()
        members = sorted(self.members, key=key)

        return members

    def __top_k_selection(self, n, k):
        members = self.__sort_members()
        top_k = members[-k:]
        parents = random.choices(top_k, k=n)

        return parents


if __name__ == '__main__':
    # RNG seed
    seed = random.random()
    random.seed(seed)
    print('Seed:', seed)

    # Experiment name and path
    experiment_name = 'tsp_1'
    os.makedirs(experiment_name, exist_ok=True)

    # Maps
    map = Map(CITIES)
    # map = Map.from_coordinates_tuples(nigeria)
    # map = Map.from_coordinates_tuples(africa)
    # map = Map.from_coordinates_tuples(world)
    
    # keep track of data
    top_best = None
    top_best_fitness = 0
    experiment_dict = {}

    for c_type in ['ox', 'pmx', 'cx']:
        # Reseed RNG each run
        random.seed(seed)

        # Restart population
        population = Population(POPULATION, map)

        # Run info
        title = c_type
        run_best_fitness = 0
        run_best_fitness_gen = 0
        run_best_history = []

        for i in range(GENERATIONS):
            best = population.generate_next_generation(c_type=c_type)
            best_fitness = best.get_fitness()

            # Check for run new best
            if best_fitness > run_best_fitness:
                run_best_fitness = best_fitness
                run_best_fitness_gen = i

                # Store new best data (distance, generation)
                run_best_history.append((round(1 / run_best_fitness, 3), i))

            # Update best solution plot every 25 generations
            if i % 25 == 0:
                plot_solution_interactive(best, generation=i, title=title)

            # If this is last generation, save plot
            if i == GENERATIONS - 1:
                filename = title + '.png'
                save_path = os.path.join(experiment_name, filename)
                plot_solution_interactive(best, generation=run_best_fitness_gen,
                                          title=title, save_path=save_path)

        # Store run data
        experiment_dict[title] = dict(best_history=run_best_history, best=best,
                                      best_fitness=run_best_fitness,
                                      best_fitness_gen=run_best_fitness_gen)

        # Check for top best
        if run_best_fitness > top_best_fitness:
            top_best_fitness = run_best_fitness
            top_best = best

        # Print run info
        print(title, 'Generation:', run_best_fitness_gen, best)
    print()
    
    # Save plots and info
    best_solution_filepath = os.path.join(experiment_name, 'best.png')
    compare_best_filepath = os.path.join(experiment_name, 'best_solutions.png')
    experiment_summary_path = os.path.join(experiment_name, 'summary.txt')

    print_experiment_summary(experiment_dict, save_path=experiment_summary_path)
    plot_solution_interactive(top_best, generation=-1, save_path=best_solution_filepath)
    plot_compare_best_history(experiment_dict, save_path=compare_best_filepath)
    
    stop_interactive_plot()
