import random


class Graph:
    def __init__(self, graph_dict, chromatic_number):
        self.chromatic_number = chromatic_number
        self.graph_dict = graph_dict


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

    def __repr__(self):
        return str(self.chromosome)