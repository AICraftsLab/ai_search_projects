import random

# Problem global variables
GENERATIONS = 1000
POPULATION = 50
GENES = 10
MUTATION_PROB = 0.01
ELITISM = 10


class Genome:
    """Class to represent a single genome"""
    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def get_fitness(self):
        return self.chromosome.count(1)
        
    def mutate(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = int(not self.chromosome[i])

    @classmethod
    def _single_point_crossover(cls, genome1_x, genome2_x):
        crossover_point = random.randrange(GENES)
        child1_x = genome1_x[:crossover_point] + genome2_x[crossover_point:]
        child2_x = genome2_x[:crossover_point] + genome1_x[crossover_point:]

        return Genome(child1_x), Genome(child2_x)

    @classmethod
    def _double_point_crossover(cls, genome1_x, genome2_x):
        crossover_points = random.sample(range(GENES), k=2)
        point1, point2 = sorted(crossover_points)

        child1_x = genome1_x[:point1] + genome2_x[point1:point2] + genome1_x[point2:]
        child2_x = genome2_x[:point1] + genome1_x[point1:point2] + genome2_x[point2:]

        return Genome(child1_x), Genome(child2_x)

    @classmethod
    def crossover(cls, genome1_x, genome2_x, c_type='single'):
        if c_type == 'single':
            return cls._single_point_crossover(genome1_x, genome2_x)
        elif c_type == 'double':
            return cls._double_point_crossover(genome1_x, genome2_x)


class Population:
    """Class to represent a population"""
    def __init__(self, size):
        self.size = size
        self.genomes = []
        self._initialize()
        
    def _initialize(self):
        """Creates the population's initial members/genomes"""
        for _ in range(self.size):
            # Creates a random chromosome of length GENES
            chromosome = [random.randrange(2) for _ in range(GENES)]
            genome = Genome(chromosome)
            self.genomes.append(genome)
    
    def _select_parents(self, n):
        """Selection process method"""
        key = lambda x: x.get_fitness()  # func to sort genomes by fitness
        genomes = sorted(self.genomes, key=key, reverse=True)  # sorts in descending order
        genomes = genomes[:POPULATION // 4]  # selects top quarter

        # randomly selects n parents from the top parents
        parents = random.choices(genomes, k=n)
        
        return parents

    def _get_elites(self, elitism):
        """Selects the elite genomes"""
        key = lambda x: x.get_fitness()  # func to sort genomes by fitness
        genomes = sorted(self.genomes, key=key, reverse=True)  # sorts in descending order
        genomes = genomes[:elitism]

        return genomes

    def generate_next_generation(self):
        """Generates the next generation's members/genomes
        by selection, crossover, mutation, and elitism"""
        next_gen_genomes = []

        parents_n = POPULATION - ELITISM
        parents = self._select_parents(parents_n)  # selection

        for i in range(0, parents_n, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            # crossover
            child1, child2 = Genome.crossover(parent1.chromosome, parent2.chromosome)
            # mutation
            child1.mutate(MUTATION_PROB)
            child2.mutate(MUTATION_PROB)
            next_gen_genomes.append(child1)
            next_gen_genomes.append(child2)

        elites = self._get_elites(ELITISM)
        next_gen_genomes.extend(elites)  # adding elites
        self.genomes = next_gen_genomes

        return elites[0]  # return best genome, the top elite


if __name__ == '__main__':
    # creates a population of size POPULATION
    population = Population(POPULATION)
    overall_best = None
    overall_best_fitness = 0
    
    for i in range(GENERATIONS):
        best = population.generate_next_generation()
        best_fitness = best.get_fitness()
        
        if overall_best is None:
            overall_best = best
            overall_best_fitness = best_fitness
        else:
            if best_fitness > overall_best_fitness:
                overall_best = best
                overall_best_fitness = best_fitness
                
        print('Generation:', i, 'Best Fitness:', overall_best_fitness, overall_best.chromosome)

        # fitness termination
        if overall_best_fitness >= GENES:
            break
        
    print('Overall Best Fitness:', overall_best_fitness, overall_best.chromosome)
