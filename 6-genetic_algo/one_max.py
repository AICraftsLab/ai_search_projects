import random


GENERATIONS = 1000
POPULATION = 50
GENES = 100
MUTATION_PROB = 0.01
ELITISM = 10


class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def get_fitness(self):
        return self.chromosome.count(1)
        
    def mutate(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = int(not self.chromosome[i])

    @classmethod
    def crossover(cls, genome1_x, genome2_x):
        crossover_point = random.randrange(GENES)
        child1_x = genome1_x[:crossover_point] + genome2_x[crossover_point:]
        child2_x = genome2_x[:crossover_point] + genome1_x[crossover_point:]
        
        return Genome(child1_x), Genome(child2_x)

class Population:
    def __init__(self, size):
        self.size = size
        self.genomes = []
        self._initialize()
        
    def _initialize(self):
        for _ in range(self.size):
            chromosome = [random.randrange(2) for _ in range(GENES)]
            genome = Genome(chromosome)
            self.genomes.append(genome)
    
    def generate_next_generation(self):
        next_gen_genomes = []
        
        parents_n = (POPULATION - ELITISM) * 2
        parents = self._select_parents(parents_n)
        
        for i in range(0, parents_n, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover(parent1.chromosome, parent2.chromosome)
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
                
        if overall_best_fitness >= GENES:
            break
        
    print('Overall Best Fitness:', overall_best_fitness, overall_best.chromosome)