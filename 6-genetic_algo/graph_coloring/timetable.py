from graph_coloring import *

# Problem global variables
GENERATIONS = 1000
POPULATION = 50


def plot_timetable(timetable_dict, solution, save_path=None):
    """Plot timetable"""
    days_n = len(timetable_dict['days'])
    periods_n = len(timetable_dict['periods'])
    day_period_n = len(timetable_dict['day_period'])

    # Timetable periods' data
    timetable_data = np.empty((day_period_n,), dtype=object)

    # Convert solution/chromosome to timetable
    for course_idx, period_idx in enumerate(solution):
        course = timetable_dict['courses'][course_idx]
        data = str(course[0]) + ':' + str(list(course[1:])) + '\n'

        if timetable_data[period_idx]:
            timetable_data[period_idx] += data
        else:
            timetable_data[period_idx] = data

    # Reshape to 2D
    timetable_data = timetable_data.reshape((days_n, periods_n))

    # Plotting
    plt.figure(figsize=(10, 6))

    # Create the table
    table = plt.table(
        cellText=timetable_data,
        colLabels=timetable_dict['periods'],
        rowLabels=timetable_dict['days'],
        loc="center",
        cellLoc="center",  # text alignment
    )

    # Increase row height
    for key, cell in table.get_celld().items():
        # key[0] represents the row index
        if key[0] > 0:  # Skip the header row
            cell.set_height(0.21)  # height value

    # Adjust column width
    table.auto_set_column_width(range(periods_n))
    table.auto_set_font_size(True)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':
    random.seed(None)

    # Create relationships between courses
    graph_dict = {
        0: [1, 5, 3, 12, 14, 15, 17],
        1: [5, 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        2: [1, 5, 6, 4, 8, 13],
        3: [1, 5, 0, 9, 12, 11, 15, 14, 17],
        4: [1, 5, 10, 2, 7, 6, 8, 12, 13, 16],
        5: [1, 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        6: [1, 5, 7, 8, 13, 10, 4, 2, 12, 16],
        7: [1, 5, 4, 6, 12, 13],
        8: [1, 5, 2, 4, 6, 13],
        9: [1, 5, 3, 11, 14],
        10: [1, 5, 4, 6, 13, 16],
        11: [1, 5, 9, 14, 3],
        12: [1, 5, 0, 3, 4, 6, 7, 13, 14, 15, 17],
        13: [1, 5, 2, 4, 6, 7, 8, 10, 16, 12],
        14: [1, 5, 0, 3, 9, 12, 11, 15, 17],
        15: [1, 5, 0, 12, 3, 14, 17],
        16: [1, 5, 4, 6, 10, 13],
        17: [1, 5, 0, 3, 12, 14, 15]
    }

    # Timetable more data
    timetable_dict = {
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'periods': ['8-10', '11-1', '1:45-3:45', '4-6'],
        'day_period': ['Mon 8-10', 'Mon 11-1', 'Mon 1:45-3:45', 'Mon 4-6',
                       'Tue 8-10', 'Tue 11-1', 'Tue 1:45-3:45', 'Tue 4-6',
                       'Wed 8-10', 'Wed 11-1', 'Wed 1:45-3:45', 'Wed 4-6',
                       'Thu 8-10', 'Thu 11-1', 'Thu 1:45-3:45', 'Thu 4-6',
                       'Fri 8-10', 'Fri 11-1', 'Fri 1:45-3:45', 'Fri 4-6',
                       ],

        # (course, students taking it) tuple
        # GROUP A: CSC, CBS, DTS
        # GROUP B: ITC, SWE, INS
        'courses': {
            0: ('ITC2203', 'ITC', 'INS'),
            1: ('GEN2203', 'ALL'),
            2: ('CYB2203', 'CBS'),
            3: ('CSC2303', 'GROUP B'),
            4: ('CSC2303', 'GROUP A'),
            5: ('GEN2201', 'ALL'),
            6: ('MTH2301', 'GROUP A'),
            7: ('CSC2201', 'CSC'),
            8: ('CYB2301', 'CBS'),
            9: ('MTH2301', 'SWE'),
            10: ('DTS2303', 'DTS'),
            11: ('SWE2305', 'SWE'),
            12: ('ITC2201', 'CSC', 'ITC', 'INS'),
            13: ('CSC2305', 'GROUP A'),
            14: ('CSC2305', 'GROUP B'),
            15: ('STA121', 'INS'),
            16: ('DTS2301', 'DTS'),
            17: ('INS2307', 'INS')
        }
    }

    # Project path
    project_name = 'timetable_1'
    os.makedirs(project_name, exist_ok=True)

    graph = Graph(graph_dict, len(timetable_dict['day_period']))
    graph_file = os.path.join(project_name, 'graph.png')
    draw_graph(graph, save_path=graph_file)

    best = None
    best_fitness = None
    population = Population(POPULATION, graph)

    for i in range(GENERATIONS):
        gen_best = population.generate_next_generation(s_type='top_k')
        gen_best_fitness = gen_best.get_fitness(graph)

        # Check for new best
        if best is None or gen_best_fitness > best_fitness:
            best = gen_best
            best_fitness = gen_best_fitness

        # Update plot every 25 generations and in the last
        if i % 25 == 0 or i + 1 == GENERATIONS:
            # Save plot in last generation
            if i + 1 == GENERATIONS:
                plot_file = os.path.join(project_name, 'coloring.png')
            else:
                plot_file = None

            plot_title = f"Timetable. Gen:{i}/{GENERATIONS} " \
                         f"Best Fitness:{best_fitness:.2f} " \
                         f"Best Colors:{best.colors} " \
                         f"Conflicts:{best.conflicts}"
            draw_graph_interactive(graph, best.chromosome, plot_title, save_path=plot_file)
        print(i, 'Fitness:', round(best_fitness, 2), best)

    timetable_file = os.path.join(project_name, 'timetable.png')
    plot_timetable(timetable_dict, best.chromosome, timetable_file)
    turn_off_interactive()
