import random
import matplotlib.pyplot as plt
import numpy as np
import os


# Problem global variables 
GENERATIONS = 10
POPULATION = 100
MUTATION_PROB = 0.01
ELITISM = 10

TOP_K_RANDOM_SELECTION = POPULATION // 4
TOURNAMENT_SIZE = POPULATION // 10
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'black']


def draw_graph(graph, show=False, save_path=None):
    plt.figure(figsize=(8, 8))  # square figure

    #vertices_pos = [(random.randrange(100), random.randrange(100)) for _ in range(graph.vertices)]
    vertices_pos = graph.vertices_pos
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
    
    plt.title(f'Graph Coloring. Vertices:{graph.vertices} Colors:{graph.chromatic_num}')
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()


def draw_graph_interactive(graph, chromosome, title, save_path=None):
    if not plt.isinteractive():
        plt.ion()

    plt.figure(1, figsize=(8, 8))  # square fig
    plt.clf()
    
    vertices_pos = graph.vertices_pos
    x_coords, y_coords = zip(*vertices_pos)
    colors = [COLORS[i] for i in chromosome]  # TODO: use try-except if indexerror to create an (RGB) tuple
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
    
    if save_path:
        plt.savefig(save_path)


def turn_off_interactive():
    plt.ioff()
    plt.show()


class Graph:
    """Class to represent graph"""
    def __init__(self, graph_dict, chromatic_number):
        self.chromatic_num = chromatic_number
        self.graph_dict = graph_dict
        self.vertices = len(self.graph_dict)
        self.vertices_pos = self._generate_vertices_pos()
        
    def _generate_vertices_pos(self):
        def divide_circle(radius, n):
            theta = 2 * np.pi / n  # Angle between points
            points = []
            r = random.random() * random.choice([1,-1])
            for i in range(n):
                x = radius * np.cos(i * theta + r)
                y = radius * np.sin(i * theta + r)
                x = round(x, 2)
                y = round(y, 2)
                points.append((x, y))
                
            return points
            
        radius = 1
        radius_incr = 3
        vertices = 3
        vertices_incr = 3
        
        pos = []
        while len(pos) < self.vertices:
            points = divide_circle(radius, vertices)
            pos.extend(points)
            if self.vertices - len(pos) > vertices_incr:
                vertices += vertices_incr
            else:
                vertices = self.vertices - len(pos)
            #print(vertices, self.vertices, self.vertices - vertices)
            radius += radius_incr
        print(len(pos))
        return pos
    
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
        #print(graph_dict)
        
        return Graph(graph_dict, chromatic_num)

class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.chromatic_num = 0
        self.conflicts = 0

    def get_fitness(self, graph):
        conflicts = 1
        for node, neighbors in graph.graph_dict.items():
            node_color = self.chromosome[node]
            neighbors_colors = [self.chromosome[i] for i in neighbors if i != node]

            if node_color in neighbors_colors:
                conflicts += 1

        unique_colors = []  # TODO: try using Counter
        for color in self.chromosome:
            if color not in unique_colors:
                unique_colors.append(color)

        self.chromatic_num = len(unique_colors)
        self.conflicts = conflicts - 1
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
        return f'Conflicts:{self.conflicts} Colors:{self.chromatic_num} {self.chromosome}'


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
        members_fitness = [m.get_fitness(self.graph) for m in self.members]
        fitness_sum = sum(members_fitness)
        probs = [x / fitness_sum for x in members_fitness]
        parents = random.choices(self.members, weights=probs, k=n)

        return parents

    def _rank_selection(self, n):
        members = self._sort_members()
        members = reversed(members)
        members = list(members)  # TODO: check the reverse operation
        
        size = len(members)
        ranks_sum = int((size + 1) * (size / 2))
        probs = [x / ranks_sum for x in range(1, size + 1)]
        parents = random.choices(members, weights=probs, k=n)

        return parents

    def _tournament_selection(self, n, t_size):
        parents = []

        def tournament(participants):
            winner = (participants[0], participants[0].get_fitness(self.graph))

            for p in participants[1:]:
                p_fitness = p.get_fitness(self.graph)
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
    random.seed(None)
    project_name = 'graph_coloring_1'
    os.makedirs(project_name, exist_ok=True)
    
    graph_dict1 = {
        0: [3],
        1: [4, 6, 3],
        2: [3],
        3: [0, 2, 5],
        4: [1, 5, 3],
        5: [3, 4, 6, 3],
        6: [1, 5, 3]
    }
    
    graph = Graph.random_graph(20, 2, 10)
    graph_file = os.path.join(project_name, 'graph.png')
    draw_graph(graph, save_path=graph_file)
    
    for s_type in ['top_k', 'roulette', 'rank', 'tournament']:
        print('Selection:', s_type)
        population = Population(POPULATION, graph)
        for i in range(GENERATIONS):
            best = population.generate_next_generation(s_type)
            best_fitness = round(best.get_fitness(graph), 2)
            best_chromatic_num = best.chromatic_num
            
            plot_title = f"S_Type:{s_type} Gen:{i+1}/{GENERATIONS} " \
                         f"Best Fitness:{best_fitness} Best Colors:{best_chromatic_num} " \
                         f"Conflicts:{best.conflicts}"
            
            if i + 1 == GENERATIONS:
                plot_file = os.path.join(project_name, f'{s_type}.png')
            else:
                plot_file = None
                
            draw_graph_interactive(graph, best.chromosome, plot_title, save_path=plot_file)
            print(i, 'Fitness', best_fitness, best)
    turn_off_interactive()
