import copy
import random


class Node:
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

    def pop(self):
        return self.nodes.pop()

    def contains(self, node):
        return any(n.state == node.state for n in self.nodes)

    def is_empty(self):
        return len(self.nodes) == 0


class Puzzle:
    def __init__(self, initial_state):
        # Initialize the puzzle with the given initial state
        self.initial_state = initial_state

        # Define the solved state of the puzzle
        self.solved_state = [[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 0]]

        # Check for duplicates and valid numbers in the initial state
        unique_numbers = []
        for row in initial_state:
            for num in row:
                # Ensure no duplicate numbers
                if num not in unique_numbers:
                    unique_numbers.append(num)
                else:
                    raise Exception("Duplicate number in puzzle")

                # Ensure numbers are between 0 and 15
                if num < 0 or num > 15:
                    raise Exception("Numbers must be between 0 and 15")

        # Ensure the puzzle has exactly 16 unique numbers
        if len(unique_numbers) != 16:
            raise Exception("Puzzle must have exactly 16 numbers (0-15)")

    def get_space(self, state):
        # Get the position of the empty space (0) in the current state
        for row_index, row in enumerate(state):
            if 0 not in row:
                continue

            space_col = row.index(0)
            space_row = row_index
            return space_row, space_col

        return None

    def actions(self, state):
        # Generate the list of possible actions from the current state
        actions_list = []

        tile_above = tile_below = tile_left = tile_right = None

        # Get the current position of the empty space (0)
        space = self.get_space(state)
        space_row = space[0]
        space_col = space[1]

        # Determine which tiles can be moved into the empty space
        if space_row - 1 >= 0:
            tile_above = state[space_row - 1][space_col]
        if space_row + 1 <= 3:
            tile_below = state[space_row + 1][space_col]
        if space_col - 1 >= 0:
            tile_left = state[space_row][space_col - 1]
        if space_col + 1 <= 3:
            tile_right = state[space_row][space_col + 1]

        # Add possible moves to the actions list
        if tile_below is not None:
            actions_list.append(('up', tile_below))
        if tile_left is not None:
            actions_list.append(('right', tile_left))
        if tile_above is not None:
            actions_list.append(('down', tile_above))
        if tile_right is not None:
            actions_list.append(('left', tile_right))

        # Shuffle actions to introduce randomness
        random.shuffle(actions_list)

        return actions_list

    def result(self, state, action):
        # Return the new state after performing the given action
        new_state = copy.deepcopy(state)
        tile_row = -1
        tile_col = -1

        # Find the position of the tile to move
        for row_index, row in enumerate(state):
            if action[1] not in row:
                continue
            tile_row = row_index
            tile_col = row.index(action[1])
            break

        # Get the current position of the empty space (0)
        space = self.get_space(state)
        space_row = space[0]
        space_col = space[1]

        # Perform the action by swapping the tile and the empty space
        new_state[space_row][space_col] = action[1]
        new_state[tile_row][tile_col] = 0

        return new_state

    def solved(self, state):
        # Check if the current state is the solved state
        return state == self.solved_state

    def get_solution(self, node):
        # Trace back from the goal node to get the solution path
        solution = []

        while node.parent:
            solution.append(node.action)
            node = node.parent

        solution.reverse()
        return solution

    def search(self):
        # Search for a solution
        frontier = StackFrontier()
        initial_node = Node(self.initial_state, None, None, 0)
        frontier.add(initial_node)

        explored = []

        while True:
            # If the frontier is empty, no solution exists
            if frontier.is_empty():
                raise Exception("No Solution")
            else:
                print(f'{len(frontier.nodes)} nodes in frontier')

            # Pop a node from the frontier
            node = frontier.pop()

            # Check if the current state is the solved state
            if self.solved(node.state):
                solution = self.get_solution(node)
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                print(solution)
                return

            # Add the current state to the explored list
            explored.append(node.state)

            # Expand the node
            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, node, action, node.cost + 1)

                # Check if the node is already in the frontier or explored
                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)


# Entry point of the script
if __name__ == "__main__":
    initial_state = [[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 0, 10, 11],
                     [13, 14, 15, 12]]

    puzzle = Puzzle(initial_state)
    puzzle.search()
