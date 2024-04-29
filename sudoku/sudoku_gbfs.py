import copy
import heapq
import math
import random


class Node:
    def __init__(self, state, action, parent, cost):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost

    def manhattan_distance(self):
        return sum([1 for row in self.state for num in row if num == 0])

    def __lt__(self, other):
        return self.manhattan_distance < other.manhattan_distance


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


class Sudoku:
    def __init__(self, initial_state):
        self.initial_state = initial_state

        rows = len(initial_state)
        cols = len(initial_state[0])

        if rows != cols:
            raise Exception("Invalid grid size: must be a perfect square")

        if not all([len(row) == cols for row in initial_state]):
            raise Exception("Invalid grid size: all rows must be equal in size")

        self.grid_size = rows
        self.box_size = int(math.isqrt(self.grid_size))

    def actions(self, state):
        next_space = None
        actions = []

        for i, row in enumerate(state):
            for j, num in enumerate(row):
                if num == 0:
                    next_space = (i, j)
                    break
            if next_space is not None:
                break

        if next_space is None:
            return actions

        for i in range(1, self.grid_size + 1):
            if self.check_validity(state, i, next_space):
                actions.append((next_space, i))

        random.shuffle(actions)
        return actions

    def check_row_validity(self, state, number, position, solved=False):
        for i, row in enumerate(state):
            if not solved:
                if i == position[0] and row.count(number) == 0:
                    return True
            else:
                if i == position[0] and row.count(number) == 1:
                    return True

        return False

    def check_col_validity(self, state, number, position, solved=False):
        col = [row[position[1]] for row in state]

        if not solved:
            return col.count(number) == 0
        else:
            return col.count(number) == 1

    def check_box_validity(self, state, number, position, solved=False):
        box_row = int(position[0] / self.box_size)
        box_col = int(position[1] / self.box_size)
        box_numbers = []

        for i, row in enumerate(state):
            if int(i / self.box_size) == box_row:
                for j, num in enumerate(row):
                    if int(j / self.box_size) == box_col:
                        box_numbers.append(num)

        if not solved:
            if box_numbers.count(number) == 0:
                return True
        else:
            if box_numbers.count(number) == 1:
                return True

        return False

    def check_validity(self, state, number, position, solved=False):
        num_row_validity = self.check_row_validity(state, number, position, solved)
        num_col_validity = self.check_col_validity(state, number, position, solved)
        num_box_validity = self.check_box_validity(state, number, position, solved)

        return all([num_row_validity, num_col_validity, num_box_validity])

    def result(self, state, action):
        new_state = copy.deepcopy(state)
        row, col = action[0][0], action[0][1]

        new_state[row][col] = action[1]
        return new_state

    def solved(self, state):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                no_zero = state[i].count(0) == 0
                if not no_zero or not self.check_validity(state, state[i][j], (i, j), solved=True):
                    return False

        return True

    def print_state(self, state):
        print()
        for row in state:
            print(row)
        print()

    def search(self):
        frontier = PriorityQueueFrontier()
        # initial_state = self.actions(self.initial_state)
        frontier.add(Node(self.initial_state, None, None, 0))

        explored = []

        while True:
            if frontier.is_empty():
                raise Exception("No solution")
            else:
                print(f'{len(frontier.nodes)} nodes in frontier')

            node = frontier.pop()

            if self.solved(node.state):
                print()
                for row in node.state:
                    print(row)

                print('Solution found')
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                return node.state

            explored.append(node.state)

            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, action, node, node.cost + 1)

                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)


if __name__ == '__main__':
    initial_state = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    sudoku = Sudoku(initial_state)
    sudoku.search()
