import heapq
import sys

import pygame
import random

pygame.init()


class Node():
    goal_state = None
    as_astar = False

    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def manhattan_distance(self):
        if Node.goal_state is None:
            raise Exception('Goal state is not assigned')

        row_diff = abs(Node.goal_state[0] - self.state[0])
        col_diff = abs(Node.goal_state[1] - self.state[1])
        distance = row_diff + col_diff

        return distance

    def __lt__(self, other):
        if Node.as_astar:
            return self.manhattan_distance() + self.cost < other.manhattan_distance() + other.cost
        else:
            return self.manhattan_distance() < other.manhattan_distance()


class StackFrontier:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def contains(self, node):
        for n in self.nodes:
            if n.state == node.state:
                return True

        return False

    def is_empty(self):
        return len(self.nodes) == 0

    def pop(self):
        return self.nodes.pop()


class QueueFrontier(StackFrontier):
    def pop(self):
        return self.nodes.pop(0)


class PriorityQueueFrontier(StackFrontier):
    def add(self, node):
        heapq.heappush(self.nodes, node)

    def pop(self):
        return heapq.heappop(self.nodes)


class Maze:
    def __init__(self, filename):
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

    def print_solved(self, solution):
        start_row, start_col = self.start

        solution_coordinates = []
        for action in solution:
            if action == 'up':
                start_row -= 1
                solution_coordinates.append((start_row, start_col))
            elif action == 'down':
                start_row += 1
                solution_coordinates.append((start_row, start_col))
            elif action == 'right':
                start_col += 1
                solution_coordinates.append((start_row, start_col))
            elif action == 'left':
                start_col -= 1
                solution_coordinates.append((start_row, start_col))

        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("#", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif (i, j) in solution_coordinates:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def actions(self, state):
        actions = []
        row, col = state

        if row - 1 >= 0 and not self.walls[row - 1][col]:
            actions.append('up')
        if row + 1 < self.height and not self.walls[row + 1][col]:
            actions.append('down')
        if col - 1 >= 0 and not self.walls[row][col - 1]:
            actions.append('left')
        if col + 1 < self.width and not self.walls[row][col + 1]:
            actions.append('right')

        random.shuffle(actions)
        return actions

    def result(self, state, action):
        row, col = state
        new_state = None

        if action == 'up':
            new_state = (row - 1, col)
        if action == 'down':
            new_state = (row + 1, col)
        if action == 'right':
            new_state = (row, col + 1)
        if action == 'left':
            new_state = (row, col - 1)

        return new_state

    def get_solution(self, node):
        solution = []

        while node.parent is not None:
            solution.append(node.action)
            node = node.parent
        solution.reverse()
        return solution


class Cell:
    size = 20
    border_width = 1

    def __init__(self, row, col, is_wall=False, is_start=False, is_goal=False):
        self.row = row
        self.col = col
        self.x = col * Cell.size
        self.y = row * Cell.size
        self.is_wall = is_wall
        self.is_goal = is_goal
        self.is_start = is_start

        self.color = 'white'
        if is_wall:
            self.color = 'dimgray'
        elif is_start:
            self.color = 'red'
        elif is_goal:
            self.color = 'blue'

    def draw(self, screen, is_current=False):
        pygame.draw.rect(screen, 'black', (self.x, self.y, Cell.size, Cell.size))
        inner_size = Cell.size - (Cell.border_width * 2)
        color = self.color if not is_current else 'green'
        pygame.draw.rect(screen, color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))

    def set_as_path(self):
        if not self.is_wall and not self.is_goal and not self.is_start:
            self.color = 'yellow'

    def set_as_explored(self):
        if not self.is_wall and not self.is_goal and not self.is_start:
            self.color = 'darkkhaki'


class MazeGame:
    def __init__(self, maze_filepath):
        with open(maze_filepath) as f:
            content = f.read()

            # Validate start and goal
            if content.count("A") != 1:
                raise Exception("maze must have exactly one start point")
            if content.count("B") != 1:
                raise Exception("maze must have exactly one goal")

        contents = content.splitlines()

        self.rows = len(contents)
        self.cols = max([len(line) for line in contents])

        self.width = self.cols * Cell.size
        self.height = self.rows * Cell.size

        self.cells = []
        self.current_cell = None

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                try:
                    if contents[i][j] == "A":
                        cell = Cell(i, j, is_start=True)
                        self.current_cell = cell
                        row.append(cell)
                    elif contents[i][j] == "B":
                        cell = Cell(i, j, is_goal=True)
                        row.append(cell)
                    elif contents[i][j] == " ":
                        cell = Cell(i, j)
                        row.append(cell)
                    else:
                        cell = Cell(i, j, is_wall=True)
                        row.append(cell)
                except IndexError:
                    cell = Cell(i, j)
                    row.append(cell)
            self.cells.append(row)

    def move(self, direction):
        row, col = self.current_cell.row, self.current_cell.col
        neighbor = None

        if direction == 'up':
            if row - 1 >= 0:
                neighbor = self.cells[row - 1][col]
        elif direction == 'down':
            if row + 1 < self.rows:
                neighbor = self.cells[row + 1][col]
        elif direction == 'right':
            if col + 1 < self.cols:
                neighbor = self.cells[row][col + 1]
        elif direction == 'left':
            if col - 1 >= 0:
                neighbor = self.cells[row][col - 1]

        if neighbor is not None and not neighbor.is_wall:
            self.current_cell = neighbor

    def solved(self):
        return self.current_cell.is_goal

    def draw(self, screen):
        for row in self.cells:
            for cell in row:
                if cell is self.current_cell:
                    cell.draw(screen, True)
                else:
                    cell.draw(screen)


def run(maze_filepath):
    maze_game = MazeGame(maze_filepath)

    maze = Maze(maze_filepath)

    Node.goal_state = maze.goal
    Node.as_astar = True
    start = Node(maze.start, None, None, 0)
    frontier = PriorityQueueFrontier()
    frontier.add(start)

    explored = []
    solution = None
    solution_index = 0

    pygame.display.set_caption('Maze Solver')
    screen = pygame.display.set_mode((maze_game.width, maze_game.height))
    clock = pygame.time.Clock()

    fps = 5

    solve_font = pygame.font.SysFont("sanscomic", int(maze_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (maze_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (maze_game.height / 2) - (solve_text.get_height() / 2)

    maze_game.draw(screen)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        if solution is not None:
            if solution_index == len(solution):
                continue

            maze_game.move(solution[solution_index])
            maze_game.current_cell.set_as_path()

            solution_index += 1

            maze_game.draw(screen)

            if maze_game.solved():
                screen.blit(solve_text, (solve_text_x, solve_text_y))

            pygame.display.flip()
            clock.tick(fps)
        else:
            # If nothing left in frontier, then no path
            if frontier.is_empty():
                raise Exception("no solution")
            else:
                print(f'{len(frontier.nodes)} nodes in frontier')
            # clock.tick(fps)
            # Choose a node from the frontier
            node = frontier.pop()

            # If node is the goal, then we have a solution
            if node.state == maze.goal:
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                solution = maze.get_solution(node)
                maze.print_solved(solution)
                print(solution)
                continue

            # Mark node as explored
            explored.append(node.state)
            maze_game.cells[node.state[0]][node.state[1]].set_as_explored()

            # Add neighbors to frontier
            for action in maze.actions(node.state):
                new_state = maze.result(node.state, action)
                new_node = Node(new_state, node, action, node.cost + 1)

                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)

            maze_game.draw(screen)

            if maze_game.solved():
                screen.blit(solve_text, (solve_text_x, solve_text_y))

            pygame.display.flip()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        maze_filepath = sys.argv[1]
        run(maze_filepath)
    else:
        maze_filepath = 'maze5.txt'
        run(maze_filepath)
