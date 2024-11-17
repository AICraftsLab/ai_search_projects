from graph_coloring3 import *

# Problem global variables
GENERATIONS = 1000
POPULATION = 50


def plot_timetable(timetable_dict, solution, save_path=None):
    """Plot timetable"""
    days_n = len(timetable_dict['days'])
    periods_n = len(timetable_dict['periods'])

    # Create day_period: (row, col) for each day-period combination
    day_period_2d_indices = {day_period: divmod(i, periods_n) for i, day_period in
                             enumerate(timetable_dict['day_period'])}

    timetable_data = np.empty((days_n, periods_n), dtype=object)

    # Convert solution/chromosome to timetable
    for course_idx, period_idx in enumerate(solution):
        period = timetable_dict['day_period'][period_idx]
        course = timetable_dict['courses'][course_idx]
        p_row, p_col = day_period_2d_indices[period]
        data = str(course[0]) + '\n' + str(list(course[1:])) + '\n'

        if timetable_data[p_row, p_col]:
            timetable_data[p_row, p_col] += data
        else:
            timetable_data[p_row, p_col] = data

    # Plotting
    plt.figure(figsize=(8, 6))

    plt.axis("off")  # Turn off the axes

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
            cell.set_height(0.22)  # Adjust height as needed

    # Style the table
    table.auto_set_font_size(True)
    table.auto_set_column_width(range(periods_n))  # Adjust column width

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
        'day_period': ['Mon 8-10am', 'Mon 11-1pm', 'Mon 1:45-3:45pm', 'Mon 4-6pm',
                       'Tue 8-10am', 'Tue 11-1pm', 'Tue 1:45-3:45pm', 'Tue 4-6pm',
                       'Wed 8-10am', 'Wed 11-1pm', 'Wed 1:45-3:45pm', 'Wed 4-6pm',
                       'Thu 8-10am', 'Thu 11-1pm', 'Thu 1:45-3:45pm', 'Thu 4-6pm',
                       'Fri 8-10am', 'Fri 11-1pm', 'Fri 1:45-3:45pm', 'Fri 4-6pm',
                       ],
        
        # (course, students taking it) tuple
        # GROUP A: CSC, CBS, DTS
        # GROUP B: ITC, SWE, INS
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
