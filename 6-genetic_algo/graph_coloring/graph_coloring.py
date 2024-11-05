import random
import matplotlib.pyplot as plt

# Problem global variables 
GENERATIONS = 1000
POPULATION = 100
MUTATION_PROB = 0.01
ELITISM = 10

TOP_K_RANDOM_SELECTION = POPULATION // 4
TOURNAMENT_SIZE = POPULATION // 10
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'black']


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


def draw_graph_interactive(graph, chromosome, vertices_pos, title):
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

    plt.title(title)
    plt.pause(0.01)


def turn_off_interactive():
    plt.ioff()
    plt.show()


class Graph:
    """Class to represent graph"""
    def __init__(self, graph_dict, chromatic_number, vertices_pos=None):
        self.chromatic_num = chromatic_number
        self.graph_dict = graph_dict
        self.vertices = len(self.graph_dict)

        if vertices_pos:
            self.vertices_pos = vertices_pos
        else:
            self.vertices_pos = [(random.randrange(100), random.randrange(100)) for _ in range(self.vertices)]

    @classmethod
    def random_graph(cls, vertices, chromatic_num, max_conn, min_conn=1):
        assert min_conn > 0 and min_conn < max_conn, "Invalid Min connections must be 0 < min_conn < max_conn"
        assert max_conn <= vertices, "Max connections must be <= vertices"
        
        graph_dict = {i: set() for i in range(vertices)}
        
        for vertex, connections in graph_dict.items():
            neighbors_n = random.randrange(min_conn, max_conn)
            while len(connections) < neighbors_n:  # TODO: bug without this line,  vertex with no connections
                sampling_pop = list(range(vertices))
                sampling_pop.remove(vertex)
                neighbors = random.sample(sampling_pop, k=neighbors_n)
                
                for neighbor in neighbors:
                    neighbor_conn = graph_dict[neighbor]
                    if len(connections) < max_conn and len(neighbor_conn) < max_conn:
                        connections.add(neighbor)
                        neighbor_conn.add(vertex)
        print(graph_dict)
        
        return Graph(graph_dict, chromatic_num)

class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.chromatic_num = 0

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
        self.chromatic_num = len(unique_colors)
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
            chromosome = [random.randrange(self.graph.chromatic_num) for x in range(self.graph.vertices)]
            self.members.append(Genome(chromosome))

    def _sort_members(self):
        key = lambda x: x.get_fitness(self.graph)
        members = sorted(self.members, key=key, reverse=True)
        
        return members

    def _top_k_random_selection(self, n, k):
        members = self._sort_members()
        top_k = members[:k]
        parents = random.choices(top_k, k=n)

        return parents

    def _roulette_selection(self, n):
        members_fitness = [m.get_fitness() for m in self.members]
        fitness_sum = sum(members_fitness)
        probs = [x / fitness_sum for x in members_fitness]
        parents = random.choices(self.members, weights=probs, k=n)

        return parents

    def _rank_selection(self, n):
        members = self._sort_members()
        members = reversed(members)
        
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
            participants = random.sample(self.members, k=t_size)
            winner = tournament(participants)
            parents.append(winner)

        return parents

    def _select_parents(self, n, s_type):
        if s_type == 'top_k':
            return self._top_k_random_selection(n, k=TOP_K_RANDOM_SELECTION)
        elif s_type == 'roulette':
            return self._roulette_selection(n)
        elif s_type == 'rank':
            return self._rank_selection(n)
        elif s_type == 'tournament':
            return self._tournament_selection(n, t_size=TOURNAMENT_SIZE)
        else:
            raise Exception('Invalid selection type')

    def generate_next_generation(self, s_type):
        next_gen_members = []

        n_parents = POPULATION - ELITISM
        parents = self._select_parents(n=n_parents, s_type=s_type)

        for i in range(0, n_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover(parent1, parent2)
            child1.mutate(MUTATION_PROB, range(self.graph.chromatic_num))
            child2.mutate(MUTATION_PROB, range(self.graph.chromatic_num))
            next_gen_members.append(child1)
            next_gen_members.append(child2)

        sorted_members = self._sort_members()
        next_gen_members.extend(sorted_members[:ELITISM])

        self.members = next_gen_members

        return sorted_members[0]


if __name__ == '__main__':
    graph_dict1 = {
        0: [3],
        1: [4, 6, 3],
        2: [3],
        3: [0, 2, 5],
        4: [1, 5, 3],
        5: [3, 4, 6, 3],
        6: [1, 5, 3]
    }
    graph_dict2 = {
        0: [3],
        1: [4, 6, 3],
        2: [3],
        3: [0, 2, 5],
        4: [1, 5, 3],
        5: [3, 4, 6, 3],
        6: [1, 5, 3],
        7: [0, 2, 4]
    }
    pos1 = [(0, 0), (10, 0), (20, 0), (10, 10), (0, 20), (10, 20), (20, 20)]
    pos2 = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    #graph = Graph(graph_dict2, 5, vertices_pos=pos2)
    graph = Graph.random_graph(10, 5, 5)
    population = Population(POPULATION, graph)
    
    for s_type in ['top_k', 'roulette', 'rank', 'tournament']:
        for i in range(GENERATIONS):
            best = population.generate_next_generation(s_type)
            best_fitness = round(best.get_fitness(graph), 2)
            best_chromatic_num = best.chromatic_num
            
            plot_title = f"S_Type:{s_type} Gen:{i}/{GENERATIONS} "\
                               f"Best Fitness:{best_fitness} Best Colors:{best_chromatic_num}"
            draw_graph_interactive(graph, best.chromosome, graph.vertices_pos, plot_title)
            print(i, best_fitness, best.chromosome, best_chromatic_num)
    turn_off_interactive()
