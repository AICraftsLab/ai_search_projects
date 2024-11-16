import random
import matplotlib.pyplot as plt
import os
import numpy as np

# Script version: Creating Graph from graph representation + ordered vertices pos

# Problem global variables
GENERATIONS = 1000
POPULATION = 50
MUTATION_PROB = 0.01
ELITISM = 10
TOP_K_SELECTION = POPULATION // 4
TOURNAMENT_SIZE = POPULATION // 10

# Create 50 random colors for the vertices
COLORS = [(random.random(), random.random(), random.random()) for _ in range(50)]


def draw_graph(graph, show=False, save_path=None):
    """Plot the graph"""
    plt.figure(figsize=(8, 8))

    # Extract and plot coordinates
    vertices_pos = graph.vertices_pos
    x_coords, y_coords = zip(*vertices_pos)
    plt.scatter(x_coords, y_coords, s=400, zorder=1)

    # Plot edges
    for vertex, neighbors in graph.graph_dict.items():
        positions = []
        vertex_pos = vertices_pos[vertex]

        # To and fro
        for neighbor in neighbors:
            neighbor_pos = vertices_pos[neighbor]
            positions.append(vertex_pos)
            positions.append(neighbor_pos)

        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, color='red', linewidth=1, zorder=0)
        plt.annotate(str(vertex), (vertex_pos[0], vertex_pos[1]), fontsize=14,
                     textcoords="offset points", xytext=(0, 15), ha='center')

    plt.title(f'Graph Coloring. Vertices:{graph.vertices} '
              f'Colors:{graph.available_colors}', fontsize=14)
    plt.axis('off')

    # If save_path is provided
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def draw_graph_interactive(graph, chromosome, title, save_path=None):
    """Interactively plots the graph evolution process"""
    # Turn on interactive mode
    if not plt.isinteractive():
        plt.ion()

    # Always draw on a single figure, fig 1
    plt.figure(1, figsize=(8, 8))
    plt.clf()  # Clear figure

    # Extract and plot vertices
    vertices_pos = graph.vertices_pos
    x_coords, y_coords = zip(*vertices_pos)
    colors = [COLORS[i] for i in chromosome]
    plt.scatter(x_coords, y_coords, c=colors, s=400, zorder=1)

    # Plot edges
    for vertex, neighbors in graph.graph_dict.items():
        positions = []
        vertex_pos = vertices_pos[vertex]

        for neighbor in neighbors:
            neighbor_pos = vertices_pos[neighbor]
            positions.append(vertex_pos)
            positions.append(neighbor_pos)

        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, color='red', linewidth=1, zorder=0)
        plt.annotate(str(vertex), (vertex_pos[0], vertex_pos[1]), fontsize=14,
                     textcoords="offset points", xytext=(0, 15), ha='center')

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.pause(0.01)  # update plot

    if save_path:
        plt.savefig(save_path)


def turn_off_interactive():
    # Turn off interactive mode and
    # redraw last plot
    plt.ioff()
    plt.show()


class Graph:
    """Class to represent graph"""
    def __init__(self, graph_dict, available_colors):
        self.available_colors = available_colors
        self.graph_dict = graph_dict
        self.vertices = len(self.graph_dict)

        # Vertices pos
        self.vertices_pos = self._generate_vertices_pos()

    def _generate_vertices_pos(self):
        """Generate vertices pos"""
        def divide_circle(radius, n):
            """Helper func to divide cycle into n points"""
            theta = 2 * np.pi / n  # Angle between points
            points = []

            # Randomly rotate the points around the origin
            r = random.random() * random.choice([1, -1])
            for i in range(n):
                x = radius * np.cos(i * theta + r)
                y = radius * np.sin(i * theta + r)
                x = round(x, 3)
                y = round(y, 3)
                points.append((x, y))

            return points

        # Starting values
        radius = 1
        vertices = 3

        # Incrementing values
        radius_incr = 3
        vertices_incr = 3

        # All vertices pos list
        pos = []

        # Generate pos
        while len(pos) < self.vertices:
            points = divide_circle(radius, vertices)
            pos.extend(points)
            radius += radius_incr
            vertices += vertices_incr
            if len(pos) + vertices > self.vertices:
                vertices = self.vertices - len(pos)

        return pos


class Genome:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.colors = 0
        self.conflicts = 0

    def get_fitness(self, graph):
        """Calculate fitness"""
        conflicts = 0
        for node, neighbors in graph.graph_dict.items():
            node_color = self.chromosome[node]
            neighbors_colors = [self.chromosome[i] for i in neighbors if i != node]

            if node_color in neighbors_colors:
                conflicts += 1

        unique_colors = []
        for color in self.chromosome:
            if color not in unique_colors:
                unique_colors.append(color)

        self.colors = len(unique_colors)
        self.conflicts = conflicts
        return -conflicts + 1 / len(unique_colors)

    def mutate(self, prob, allele):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = random.choice(allele)

    @classmethod
    def crossover(cls, parent1, parent2):
        """Single-point crossover"""
        point = random.randrange(len(parent1.chromosome))

        chromosome1 = parent1.chromosome[:point] + parent2.chromosome[point:]
        chromosome2 = parent2.chromosome[:point] + parent1.chromosome[point:]

        offspring1 = Genome(chromosome1)
        offspring2 = Genome(chromosome2)

        return offspring1, offspring2

    def __repr__(self):
        """String representation of genome"""
        return f'Conflicts:{self.conflicts} Colors:{self.colors} {self.chromosome}'


class Population:
    def __init__(self, size, graph):
        self.size = size
        self.graph = graph
        self.genomes = []

        self._initialize()

    def _initialize(self):
        """Initialize the population"""
        for i in range(self.size):
            chromosome = [random.randrange(self.graph.available_colors)
                          for x in range(self.graph.vertices)]
            self.genomes.append(Genome(chromosome))

    def _sort_members(self):
        """Sort members in descending order"""
        key = lambda x: x.get_fitness(self.graph)
        members = sorted(self.genomes, key=key, reverse=True)

        return members

    def _top_k_selection(self, n, k):
        """Select n parents from the top k genomes"""
        genomes = self._sort_members()
        top_k = genomes[:k]
        parents = random.choices(top_k, k=n)

        return parents

    def _roulette_selection(self, n):
        genomes_fitness = [m.get_fitness(self.graph) for m in self.genomes]
        fitness_sum = sum(genomes_fitness)
        probabilities = [x / fitness_sum for x in genomes_fitness]
        parents = random.choices(self.genomes, weights=probabilities, k=n)

        return parents

    def _rank_selection(self, n):
        genomes = self._sort_members()  # Descending
        genomes = reversed(genomes)  # Ascending
        genomes = list(genomes)

        size = len(genomes)
        ranks_sum = int((size + 1) * (size / 2))  # using Gauss summation
        probabilities = [x / ranks_sum for x in range(1, size + 1)]
        parents = random.choices(genomes, weights=probabilities, k=n)

        return parents

    def _tournament_selection(self, n, t_size):
        parents = []

        def tournament(participants):
            """Perform tournament and return winner"""
            winner = (participants[0], participants[0].get_fitness(self.graph))

            for p in participants[1:]:
                p_fitness = p.get_fitness(self.graph)
                if p_fitness > winner[1]:
                    winner = (p, p_fitness)

            return winner[0]

        while len(parents) < n:
            participants = random.sample(self.genomes, k=t_size)
            winner = tournament(participants)
            parents.append(winner)

        return parents

    def _select_parents(self, n, s_type):
        """Perform a specific selection operation"""
        if s_type == 'top_k':
            return self._top_k_selection(n, k=TOP_K_SELECTION)
        elif s_type == 'roulette':
            return self._roulette_selection(n)
        elif s_type == 'rank':
            return self._rank_selection(n)
        elif s_type == 'tournament':
            return self._tournament_selection(n, t_size=TOURNAMENT_SIZE)
        else:
            raise Exception('Invalid selection type')

    def generate_next_generation(self, s_type):
        """Perform the evolution process"""
        next_gen_genomes = []

        # Selection
        n_parents = POPULATION - ELITISM
        parents = self._select_parents(n=n_parents, s_type=s_type)

        # Crossover and mutation
        for i in range(0, n_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = Genome.crossover(parent1, parent2)
            child1.mutate(MUTATION_PROB, range(self.graph.available_colors))
            child2.mutate(MUTATION_PROB, range(self.graph.available_colors))
            next_gen_genomes.append(child1)
            next_gen_genomes.append(child2)

        # Elitism
        sorted_members = self._sort_members()
        next_gen_genomes.extend(sorted_members[:ELITISM])

        self.genomes = next_gen_genomes

        return sorted_members[0]


if __name__ == '__main__':
    # Seeding for reproducibility
    random.seed(30)

    # Project path
    project_name = 'graph_coloring_1'
    os.makedirs(project_name, exist_ok=True)

    # Graph representation
    graph_dict = {
        0: [3, 4, 5],  # A
        1: [2, 3],  # B
        2: [4, 5],  # C
        3: [0, 1],  # D
        4: [0, 2, 5],  # E
        5: [0, 2, 4]  # F
    }

    # Creating Graph using graph dict
    graph = Graph(graph_dict, 5)
    graph_file = os.path.join(project_name, 'graph.png')
    draw_graph(graph, save_path=graph_file)  # Plot graph

    for s_type in ['top_k', 'roulette', 'rank', 'tournament']:
        print('Selection:', s_type)
        population = Population(POPULATION, graph)
        best = None
        best_fitness = None

        for i in range(GENERATIONS):
            gen_best = population.generate_next_generation(s_type)
            gen_best_fitness = gen_best.get_fitness(graph)

            # Check for new best
            if best is None or gen_best_fitness > best_fitness:
                best = gen_best
                best_fitness = gen_best_fitness

            # Save plot in last generation
            if i + 1 == GENERATIONS:
                plot_file = os.path.join(project_name, f'{s_type}.png')
            else:
                plot_file = None

            # Update plot every 25 generations and in the last
            if i % 25 == 0 or i + 1 == GENERATIONS:
                plot_title = f"S_Type:{s_type} Gen:{i}/{GENERATIONS} " \
                             f"Best Fitness:{best_fitness:.2f} " \
                             f"Best Colors:{best.colors} " \
                             f"Conflicts:{best.conflicts}"
                draw_graph_interactive(graph, best.chromosome, plot_title, save_path=plot_file)
            print(i, 'Fitness', round(best_fitness, 2), best)

    turn_off_interactive()
