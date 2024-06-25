import copy
import random
from node import Node
from frontier import StackFrontier


class Sudoku:
    def __init__(self, initial_state):
        self.initial_state = initial_state

        # Define Grid (9x9) and subgrid (3x3) sizes
        self.grid_size = 9
        self.box_size = 3

        # Validate that the provided grid is 9x9
        rows = len(initial_state)
        cols = len(initial_state[0])

        if rows != self.grid_size or cols != self.grid_size:
            raise Exception("Invalid grid size: must be 9x9")

        if not all(len(row) == self.grid_size for row in initial_state):
            raise Exception("Invalid grid size: all rows must be equal in size")

    def actions(self, state):
        # Determine the possible actions (valid numbers for the next empty cell)
        next_space = None
        actions = []

        # Find the next empty space (represented by 0)
        for i, row in enumerate(state):
            for j, num in enumerate(row):
                if num == 0:
                    next_space = (i, j)
                    break
            if next_space is not None:
                break

        # If there's no empty space, return an empty list (puzzle is complete)
        if next_space is None:
            return actions

        # Check each number from 1 to 9 to see if it is a valid action
        for i in range(1, self.grid_size + 1):
            if self.check_validity(state, i, next_space):
                actions.append((next_space, i))

        # Shuffle actions to introduce randomness
        random.shuffle(actions)
        return actions

    def check_validity(self, state, number, position, solved=False):
        # Check if placing a number at a given position is valid
        num_row_validity = self.check_row_validity(state, number, position, solved)
        num_col_validity = self.check_col_validity(state, number, position, solved)
        num_box_validity = self.check_box_validity(state, number, position, solved)

        return all([num_row_validity, num_col_validity, num_box_validity])

    def check_row_validity(self, state, number, position, solved=False):
        # Validate number in the row at a position
        for i, row in enumerate(state):
            if not solved:
                if i == position[0] and row.count(number) == 0:
                    return True
            else:
                if i == position[0] and row.count(number) == 1:
                    return True

        return False

    def check_col_validity(self, state, number, position, solved=False):
        # Validate number in the column at a position

        # Getting the numbers at the position across all rows
        col = [row[position[1]] for row in state]

        if not solved:
            return col.count(number) == 0
        else:
            return col.count(number) == 1

    def check_box_validity(self, state, number, position, solved=False):
        # Validate number in the sub-grid (3x3 box)

        # Get the row and col of the box of the number
        box_row = int(position[0] / self.box_size)
        box_col = int(position[1] / self.box_size)

        box_numbers = []

        # Get only numbers inside the box
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

    def result(self, state, action):
        # Return a new state after applying the action (placing a number)

        # Create a deep copy of the current state to avoid changing it
        new_state = copy.deepcopy(state)

        # Get the position to place the number
        row, col = action[0][0], action[0][1]

        # Place the number in the specified position
        new_state[row][col] = action[1]

        return new_state

    def solved(self, state):
        # Check if the current state is the solved state
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Check if there are no zeros in the row
                no_zero = state[i].count(0) == 0

                # Ensure the current number is valid in its row, column, and box
                valid = self.check_validity(state, state[i][j], (i, j), solved=True)

                if not no_zero or not valid:
                    return False

        return True

    def print_state(self, state):
        # Print the current state of the Sudoku puzzle
        print()
        for row in state:
            print(row)
        print()

    def search(self):
        # Search for the solution
        frontier = StackFrontier()
        frontier.add(Node(self.initial_state, None, None, 0))

        explored = []

        while True:
            # If the frontier is empty, no solution exists
            if frontier.is_empty():
                raise Exception("No solution")
            else:
                print(f'{len(frontier.nodes)} nodes in frontier')

            # Pop a node from the frontier
            node = frontier.pop()

            # Check if the current state is the solved state
            if self.solved(node.state):
                self.print_state(node.state)
                print('Solution found')
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                return

            # Add the current state to the explored list
            explored.append(node.state)

            # Expand the node
            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, action, node, node.cost + 1)

                # Check if the node is already in the frontier or explored
                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)


# Entry point of the script
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
