import random
import matplotlib.pyplot as plt


GENERATIONS = 500
POPULATION = 100
TOTAL_ITEMS = 20
BAG_CAPACITY = 1000
MUTATION_PROB = 0.01
ELITISM = 10


def plot_best_history(best_history, save_path=None):
    plt.figure()
    
    x_coord, values, weights = zip(*best_history)
    
    y_coord = values
    plt.plot(x_coord, y_coord, label='Values')
    
    y_coord = weights
    plt.plot(x_coord, y_coord, label='Capacity')
    
    plt.title('Knapsack best history')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Value')
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def print_items(items):
    for item in items:
        print(item)


class Item:
    def __init__(self, weight, value, name):
        self.weight = weight
        self.value = value
        self.name = name
        
    def __repr__(self):
        return f'Item {self.name} Value:{self.value} Weight:{self.weight}'
        
    @classmethod
    def from_random_items(cls, n, min_value=10, max_value=100, min_weight=10, max_weight=BAG_CAPACITY // 5):
        items = []
        
        for i in range(n):
            value = random.randrange(min_value, max_value)
            weight = random.randrange(min_weight, max_weight)
            name = str(i)
            
            item = Item(weight, value, name)
            items.append(item)
            
        return items


class Genome:
    ITEMS = []
    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def get_fitness(self):
        fitness = 0
        total_weight = 0
        
        for i in range(len(self.chromosome)):
            gene = self.chromosome[i]
            if gene == 1:
                item = self.ITEMS[i]
                fitness += item.value
                total_weight += item.weight
        
        if total_weight > BAG_CAPACITY:
            fitness = 0
            
        return fitness
    
    def get_weight(self):
        total_weight = 0
        
        for i in range(len(self.chromosome)):
            gene = self.chromosome[i]
            if gene == 1:
                item = self.ITEMS[i]
                total_weight += item.weight
        
        return total_weight
    
    def mutate(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = int(not self.chromosome[i])
    
    def print_items(self):
        for i, gene in enumerate(self.chromosome):
            if gene == 1:
                item = self.ITEMS[i]
                print(item)
    
    @classmethod
    def crossover(cls, genome1_x, genome2_x):
        crossover_point = random.randrange(TOTAL_ITEMS)
        child1_x = genome1_x[:crossover_point] + genome2_x[crossover_point:]
        child2_x = genome2_x[:crossover_point] + genome1_x[crossover_point:]
        
        return Genome(child1_x), Genome(child2_x)
        
    @classmethod
    def crossover_2p(cls, genome1_x, genome2_x):
        points = random.sample(range(TOTAL_ITEMS), k=2)
        points.sort()
        point1, point2 = points
        child1_x = genome1_x[:point1] + genome2_x[point1:point2] + genome1_x[point2:]
        child2_x = genome2_x[:point1] + genome1_x[point1:point2] + genome2_x[point2:]
        
        return Genome(child1_x), Genome(child2_x)
        

class Population:
    def __init__(self, size):
        self.size = size
        self.genomes = []
        self._initialize()
        
    def _initialize(self):
        for _ in range(self.size):
            chromosome = [random.randrange(1) for _ in range(TOTAL_ITEMS)]
            genome = Genome(chromosome)
            self.genomes.append(genome)
    
    def generate_next_generation(self):
        next_gen_genomes = []
        
        parents_n = (POPULATION - ELITISM) * 2
        parents = self._select_parents(parents_n)
        
        for i in range(0, parents_n, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover_2p(parent1.chromosome, parent2.chromosome)
            child1.mutate(MUTATION_PROB)
            child2.mutate(MUTATION_PROB)
            next_gen_genomes.append(child1)
            next_gen_genomes.append(child2)
            
        elites = self._get_elites(ELITISM)
        next_gen_genomes.extend(elites)
        self.genomes = next_gen_genomes
        
        return elites[0]
    
    def _get_elites(self, elitism):
        genomes = [(g, g.get_fitness()) for g in self.genomes]
        key = lambda x: x[1]
        genomes.sort(key=key, reverse=True)
        genomes = genomes[:elitism]
        genomes = [g[0] for g in genomes]
        
        return genomes
    
    def _select_parents(self, n):
        genomes = [(g, g.get_fitness()) for g in self.genomes]
        key = lambda x: x[1]
        genomes.sort(key=key, reverse=True)
        genomes = genomes[:POPULATION // 4]
        genomes = [g[0] for g in genomes]
        
        parents = random.choices(genomes, k=n)
        
        return parents
        

if __name__ == '__main__':
    seed = None
    random.seed(seed)
    
    population = Population(POPULATION)
    items = Item.from_random_items(TOTAL_ITEMS)
    Genome.ITEMS = items
    
    overall_best = None
    overall_best_fitness = 0
    best_history = []
    
    for i in range(GENERATIONS):
        best = population.generate_next_generation()
        best_fitness = best.get_fitness()
        
        if overall_best is None:
            overall_best = best
            overall_best_fitness = best_fitness
            best_history.append((i, best_fitness, best.get_weight()))
        else:
            if best_fitness > overall_best_fitness:
                overall_best = best
                overall_best_fitness = best_fitness
                best_history.append((i, best_fitness, best.get_weight()))
                
        print('Generation:', i, 'Best Fitness:', overall_best_fitness, overall_best.chromosome)
    
    print('All items')
    print_items(items)
    print()
    
    print('Overall Best Fitness:', overall_best_fitness, overall_best.chromosome)
    overall_best.print_items()
    print('Total weight', overall_best.get_weight())
    
    best_history.append((i, best_fitness, best.get_weight()))
    plot_best_history(best_history)