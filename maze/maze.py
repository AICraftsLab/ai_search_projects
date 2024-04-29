import sys
import random


class Node():
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost


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


class Maze():

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

    def solve(self):
        start = Node(self.start, None, None, 0)
        frontier = StackFrontier()
        frontier.add(start)

        explored = []

        while True:
            # If nothing left in frontier, then no path
            if frontier.is_empty():
                raise Exception("no solution")
            else:
                print(f'{len(frontier.nodes)} nodes in frontier')

            # Choose a node from the frontier
            node = frontier.pop()

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                solution = self.get_solution(node)
                self.print_solved(solution)
                print(solution)
                return

            # Mark node as explored
            explored.append(node.state)

            # Add neighbors to frontier
            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, node, action, node.cost + 1)

                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        maze = Maze(sys.argv[1])
        maze.solve()
    else:
        maze = Maze('maze2.txt')
        maze.solve()

