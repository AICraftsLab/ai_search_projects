import random
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

GENERATIONS = 100
POPULATION = 100
MUTATION_PROB = 0.01
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
    #plt.show()


def create_timetable(timetable_dict, solution):
    timetable = {period: [] for period in timetable_dict['periods']}

    for course_idx, period_idx in enumerate(solution):
        period = timetable_dict['days_periods'][period_idx]
        course = timetable_dict['courses'][course_idx]

        timetable[period].append(course)

    return timetable

def plot_timetable(timetable_dict, solution):
    days_n = len(timetable_dict['days'])
    periods_n = len(timetable_dict['periods'])
    days_periods_2d_indices = {day_period: divmod(i, periods_n) for i, day_period in enumerate(timetable_dict['days_periods'])}
    timetable_data = np.full((days_n, periods_n, 1), '')
    
    for course_idx, period_idx in enumerate(solution):
        period = timetable_dict['days_periods'][period_idx]
        course = timetable_dict['courses'][course_idx]
        p_row, p_col = days_periods_2d_indices[period]
        print(course, p_row, p_col, period)
        data = str(course[0]) + '\n' + str(course[1:])
        timetable_data[p_row][p_col].append(data)

    return timetable_data

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
        return 3 / conflicts #+ 1 / len(unique_colors)

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

    timetable_dict = {
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'periods': ['8-10', '11-1', '1:45-3:45', '4-6'],
        'days_periods': ['Mon 8-10am', 'Mon 11-1pm', 'Mon 1:45-3:45pm', 'Mon 4-6pm',
                    'Tue 8-10am', 'Tue 11-1pm', 'Tue 1:45-3:45pm', 'Tue 4-6pm',
                    'Wed 8-10am', 'Wed 11-1pm', 'Wed 1:45-3:45pm', 'Wed 4-6pm',
                    'Thu 8-10am', 'Thu 11-1pm', 'Thu 1:45-3:45pm', 'Thu 4-6pm',
                    'Fri 8-10am', 'Fri 11-1pm', 'Fri 1:45-3:45pm', 'Fri 4-6pm',
                    ],
        'courses': {0: ('ITC2203', 'ITC', 'INS'),
                    1: ('GEN2203', 'ALL'),
                    2: ('CYB2203', 'CBS'),
                    3: ('CSC2303', 'GROUP B'),
                    4: ('CSC2303', 'GROUP A'),
                    5: ('GEN2201', 'ALL'),
                    6: ('MTH2301', 'GROUP A'),
                    7: ('CSC2201', 'CSC'),
                    8: ('CYB2301', 'CBS'),
                    9: ('MTH2301', 'GROUP B'),
                    10: ('DTS2303', 'DTS'),
                    11: ('SWE2305', 'SWE'),
                    12: ('ITC2201', 'CSC', 'ITC', 'CBS', 'INS'),
                    13: ('CSC2305', 'GROUP A'),
                    14: ('CSC2305', 'GROUP B'),
                    15: ('STA121', 'INS'),
                    16: ('DTS2301', 'DTS')
                    }
    }

    graph = Graph(graph_dict, 19)
    population = Population(POPULATION, graph)

    for i in range(GENERATIONS):
        best = population.generate_next_generation()
        
        if i % 25 == 0 or i == GENERATIONS - 1:
            draw_graph_interactive(graph, best.chromosome, i, graph.vertices_pos)
            #print(i, round(best.get_fitness(graph), 2), best.chromosome, best.num)

    timetable = plot_timetable(timetable_dict, best.chromosome)
    pprint(timetable)
    turn_off_interactive()
