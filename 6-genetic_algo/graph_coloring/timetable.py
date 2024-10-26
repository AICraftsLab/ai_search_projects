import random

import matplotlib.pyplot as plt

GENERATIONS = 5000
POPULATION = 100
MUTATION_PROB = 0.1
ELITISM = 10

TOP_K_RANDOM_SELECTION = POPULATION // 4
TOURNAMENT_SIZE = POPULATION // 10
COLORS = [(random.random(), random.random(), random.random()) for _ in range(50)]



def draw_graph(graph, vertices_pos=None):
    plt.figure()

    if not vertices_pos:
        vertices_pos = [(random.randrange(100), random.randrange(100)) for _ in range(graph.vertices)]
    x_coords, y_coords = zip(*vertices_pos)
    plt.scatter(x_coords, y_coords, s=100, zorder=1)

    for vertex, neighbors in graph.graph_dict.items():
        positions = []
        vertex_pos = vertices_pos[vertex]

        for neighbor in neighbors:
            neighbor_pos = vertices_pos[neighbor]
            positions.append(vertex_pos)
            positions.append(neighbor_pos)

        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, color='red', linewidth=1, zorder=0)
        plt.annotate(str(vertex), (vertex_pos[0], vertex_pos[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()


def draw_graph_interactive(graph, chromosome, generation, vertices_pos):
    if not plt.isinteractive():
        plt.ion()

    plt.figure(1)
    plt.clf()

    x_coords, y_coords = zip(*vertices_pos)
    colors = [COLORS[i] for i in chromosome]
    plt.scatter(x_coords, y_coords, c=colors, s=100, zorder=1)

    for vertex, neighbors in graph.graph_dict.items():
        positions = []
        vertex_pos = vertices_pos[vertex]

        for neighbor in neighbors:
            neighbor_pos = vertices_pos[neighbor]
            positions.append(vertex_pos)
            positions.append(neighbor_pos)

        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, color='red', linewidth=1, zorder=0)
        plt.annotate(str(vertex), (vertex_pos[0], vertex_pos[1]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title('Time Table. Generation: ' + str(generation))
    
    plt.pause(0.01)


def turn_off_interactive():
    plt.ioff()
    plt.show()


class Graph:
    def __init__(self, graph_dict, chromatic_number, vertices_pos=None):
        self.chromatic_number = chromatic_number
        self.graph_dict = graph_dict
        self.vertices = len(self.graph_dict)

        if vertices_pos:
            self.vertices_pos = vertices_pos
        else:
            self.vertices_pos = [(random.randrange(100), random.randrange(100)) for _ in range(self.vertices)]


class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.num = 0

    def get_fitness(self, graph):
        conflicts = 1
        for node, neighbors in graph.graph_dict.items():
            node_color = self.chromosome[node]
            neighbors_colors = [self.chromosome[i] for i in neighbors if i != node]

            if node_color in neighbors_colors:
                conflicts += 1

        unique_colors = []
        for color in self.chromosome:
            if color not in unique_colors:
                unique_colors.append(color)
        self.num = len(unique_colors)
        return 3 / conflicts + 1 / len(unique_colors)

    def mutate(self, prob, allele):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = random.choice(allele)

    @classmethod
    def crossover(cls, parent1, parent2):
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

        self._initialize()

    def _initialize(self):
        for i in range(self.size):
            chromosome = [random.randrange(self.graph.chromatic_number) for x in range(self.graph.vertices)]
            self.members.append(Genome(chromosome))

    def _top_k_random_selection(self, n, k):
        key = lambda x: x.get_fitness(self.graph)
        members = sorted(self.members, key=key, reverse=True)
        members = members[:k]

        parents = random.choices(members, k=n)
        return parents

    def generate_next_generation(self):
        generation_members = []

        n_parents = POPULATION * 2 - ELITISM
        parents = self._top_k_random_selection(n=n_parents, k=TOP_K_RANDOM_SELECTION)

        for i in range(0, n_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover(parent1, parent2)
            child1.mutate(MUTATION_PROB, range(self.graph.chromatic_number))
            child2.mutate(MUTATION_PROB, range(self.graph.chromatic_number))
            generation_members.append(child1)
            generation_members.append(child2)

        key = lambda x: x.get_fitness(self.graph)
        sorted_members = sorted(self.members, key=key, reverse=True)
        generation_members.extend(sorted_members[:ELITISM])

        self.members = generation_members

        return sorted_members[0]


if __name__ == '__main__':
    graph_dict = {
        1: [2, 6, 4, 13, 15, 16],
        2: [6, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        3: [2, 6, 7, 5, 9, 13, 14],
        4: [2, 6, 13, 12, 16, 10, 15, 1],
        5: [2, 6, 11, 3, 8, 7, 9, 13, 14, 17],
        6: [2, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        7: [2, 6, 8, 9, 14, 11, 5, 3, 13, 17],
        8: [2, 6, 5, 7, 13, 14],
        9: [2, 6, 3, 5, 7, 13, 14],
        10: [2, 6, 4, 12, 15],
        11: [2, 6, 5, 7, 14, 17],
        12: [2, 6, 10, 15, 4],
        13: [2, 6, 1, 3, 4, 5, 7, 8, 9, 14, 15, 16],
        14: [2, 6, 3, 5, 7, 8, 9, 11, 17, 13],
        15: [2, 6, 1, 4, 10, 13, 12, 16],
        16: [2, 6, 1, 13, 4, 15],
        17: [2, 6, 5, 7, 11, 14],
    }
    
    graph_dict = {
        0: [1, 5, 3, 12, 14, 15],
        1: [5, 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        2: [1, 5, 6, 4, 8, 12, 13],
        3: [1, 5, 12, 11, 15, 9, 14, 0],
        4: [1, 5, 10, 2, 7, 6, 8, 12, 13, 16],
        5: [1, 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        6: [1, 5, 7, 8, 13, 10, 4, 2, 12, 16],
        7: [1, 5, 4, 6, 12, 13],
        8: [1, 5, 2, 4, 6, 12, 13],
        9: [1, 5, 3, 11, 14],
        10: [1, 5, 4, 6, 13, 16],
        11: [1, 5, 9, 14, 3],
        12: [1, 5, 0, 2, 3, 4, 6, 7, 8, 13, 14, 15],
        13: [1, 5, 2, 4, 6, 7, 8, 10, 16, 12],
        14: [1, 5, 0, 3, 9, 12, 11, 15],
        15: [1, 5, 0, 12, 3, 14],
        16: [1, 5, 4, 6, 10, 13],
    }
#    d = {}
#    for node, neighbors in graph_dict.items():
#        n_node = node - 1
#        l = []
#        for n in neighbors:
#            l.append(n - 1)
#        d[n_node] = l
#        
#    print(d)
#    quit()
    #pos = [(0, 0), (10, 0), (20, 0), (10, 10), (0, 20), (10, 20), (20, 20)]
    graph = Graph(graph_dict, 19)
    population = Population(POPULATION, graph)

    for i in range(GENERATIONS):
        best = population.generate_next_generation()
        
        if i % 25 == 0 or i == GENERATIONS - 1:
            draw_graph_interactive(graph, best.chromosome, i, graph.vertices_pos)
            print(i, round(best.get_fitness(graph), 2), best.chromosome, best.num)
    turn_off_interactive()
