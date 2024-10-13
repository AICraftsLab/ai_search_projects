import random


class Graph:
    def __init__(self, graph_dict, chromatic_number):
        self.chromatic_number = chromatic_number
        self.graph_dict = graph_dict
        self.vertices = len(self.graph_dict)


class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def get_fitness(self, graph):
        conflicts = 1
        for node, neighbors in graph.items():
            node_color = self.chromosome[node]
            neighbors_colors = [self.chromosome[i] for i in neighbors if i != node]

            if node_color in neighbors_colors:
                conflicts += 1

        return 1 / conflicts

    def mutate(self, prob, allele):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = random.choice(allele)

    @classmethod
    def reproduce(cls, parent1, parent2):
        locus = random.randrange(len(parent1.chromosome))

        chromosome1 = parent1.chromosome[:locus] + parent2.chromosome[locus:]
        chromosome2 = parent2.chromosome[:locus] + parent1.chromosome[locus:]

        offspring1 = Genome(chromosome1)
        offspring2 = Genome(chromosome2)

        return offspring1, offspring2

    def __repr__(self):
        return str(self.chromosome)


class Population:
    def __init__(self, size, graph):
        self.size = size
        self.graph = graph
        self.members = []

    def _initialize(self):
        for i in range(self.size):
            chromosome = [random.randrange(self.graph.chromatic_number) for x in range(self.graph.vertices)]
            self.members.append(Genome(chromosome))

    def _select_best(self, n):
        pass