import os.path
import random
import math
import matplotlib.pyplot as plt
import pickle

from custom_maps import nigeria, africa, world


# Problem global variables 
GENERATIONS = 2000
CITIES = 30
MAP_SIZE = (300, 300)
POPULATION = 100
ELITISM = 10
TOP_K = POPULATION // 4


def plot_solution_interactive(genome, generation, title='', save_path=None):
    # Get figure number 1 or create it if not exist
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
    plt.scatter(x_coords, y_coords, color='blue', s=100, zorder=1)

    # Annotate the cities with their names
    for i, name in enumerate(names):
        plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot the path between cities
    x_coords = list(x_coords) + [x_coords[0]]
    y_coords = list(y_coords) + [y_coords[0]]
    plt.plot(x_coords, y_coords , color='red', zorder=0)

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'TSP: {len(cities)} cities. {title} Generation:{generation} Best:{1/genome.get_fitness():.2f}')

    # Pause to update the plot
    plt.pause(0.1)
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)


def plot_best_history(best_history, generations, save_path=None):
    """Func to compare best genomes gotten overtime"""
    plt.figure(figsize=(10, 7))

    y_coords, x_coords = zip(*best_history)
    
    x_coords = list(x_coords)
    x_coords.append(generations)
    
    y_coords = list(y_coords)
    y_coords.append(y_coords[-1])
    
    plt.plot(x_coords, y_coords)

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
    #def _swap_mutation(self):
        points = random.sample(range(len(self.chromosome)), k=2)
        p1, p2 = points 
        self.chromosome[p1], self.chromosome[p2] = self.chromosome[p2], self.chromosome[p1]

    def get_solution(self):
        """Return a list of city indices in chromosome"""
        return [city.idx for city in self.chromosome]

    def __repr__(self):
        """String representation of genome"""
        return f'Distance:{1/self.get_fitness():.2f} Solution:{self.get_solution()}'

            
class Population:
    def __init__(self, size, map):
        self.size = size
        self.map = map
        self.members = []
        self._initialize()

    def _initialize(self):
        for i in range(self.size):
            genome = Genome(random.sample(self.map.cities, self.map.cities_n))
            self.members.append(genome)

    def generate_next_generation(self):
        next_gen_members = []
        parents_n = self.size - ELITISM
        parents = self._top_k_random_selection(parents_n, TOP_K)

        for i in range(parents_n):
            parent = parents[i]
            child = parent.replicate()
            child.mutate()
            next_gen_members.append(child)

        sorted_members = self._sort_members()
        next_gen_members.extend(sorted_members[-ELITISM:])

        self.members = next_gen_members

        return sorted_members[-1]

    def _sort_members(self):
        key = lambda x: x.get_fitness()
        members = sorted(self.members, key=key)

        return members

    def _top_k_random_selection(self, n, k):
        members = self._sort_members()
        top_k = members[-k:]
        parents = random.choices(top_k, k=n)

        return parents


if __name__ == '__main__':
    # RNG seed
    map_seed = 32
    solution_seed = None
    random.seed(map_seed)
    
    # Experiment name and path
    experiment_name = 'tmp'
    os.makedirs(experiment_name, exist_ok=True)

    # Maps
    map = Map(CITIES)
    # map = Map.from_coordinates_tuples(nigeria)
    # map = Map.from_coordinates_tuples(africa)
    # map = Map.from_coordinates_tuples(world)
    
    # keep track of data
    top_best_fitness = 0
    top_best_fitness_gen = 0
    top_best_history = []
    
    if True:
        if True:
            # Seed solution RNG
            random.seed(solution_seed)
            
            # Create population
            population = Population(POPULATION, map)
            
            for i in range(GENERATIONS):
                best = population.generate_next_generation()
                best_fitness = best.get_fitness()
                
                # Check fornew best
                if best_fitness > top_best_fitness:
                    top_best_fitness = best_fitness
                    top_best_fitness_gen = i
                    
                    # Store new best data (distance, generation)
                    top_best_history.append((round(1 / top_best_fitness, 3), i))

                # Update best solution plot every 25 generations
                if i % 25 == 0:
                    plot_solution_interactive(best, generation=i, title='Replication')

                # If this is last generation, save plot
                if i == GENERATIONS - 1:
                    filename = 'best.png'
                    save_path = os.path.join(experiment_name, filename)
                    plot_solution_interactive(best, generation=top_best_fitness_gen, title='Replication', save_path=save_path)

                # Print info
                #print('Generation:', i, 'Best info:', 'Gen:', top_best_fitness_gen, best)
            print()
    
    # Plots best history
    best_history_filepath = os.path.join(experiment_name, 'best_history.png')
    plot_best_history(top_best_history, GENERATIONS, save_path=best_history_filepath)
    
    stop_interactive_plot()
