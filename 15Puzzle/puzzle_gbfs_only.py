import copy
import random
import heapq


class Node:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def manhattan_distance(self):
        distance = 0
        for i, row in enumerate(self.state):
            for j, num in enumerate(row):
                for k, goal_state_row in enumerate(Puzzle.solved_state):
                    if num in goal_state_row:
                        l = goal_state_row.index(num)
                        row_diff = abs(i - k)
                        col_diff = abs(j - l)
                        distance += row_diff + col_diff
                        break

        return distance

    def __lt__(self, other):
        return self.manhattan_distance() < other.manhattan_distance()


class PriorityQueueFrontier:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        heapq.heappush(self.nodes, node)

    def pop(self):
        return heapq.heappop(self.nodes)

    def contains(self, node):
        for n in self.nodes:
            if n.state == node.state:
                return True

        return False

    def is_empty(self):
        return len(self.nodes) == 0


class Puzzle:
    solved_state = [[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 0]]

    def __init__(self, initial_state):
        self.initial_state = initial_state

        unique_numbers = []
        for row in initial_state:
            for num in row:
                if num not in unique_numbers:
                    unique_numbers.append(num)
                else:
                    raise Exception("Duplicate number in puzzle")

                if num < 0 or num > 15:
                    raise Exception("Numbers must be between 0 and 15")

        if len(unique_numbers) != 16:
            raise Exception("Puzzle must have exactly 16 numbers (0-15)")

    def actions(self, state):
        actions_list = []

        tile_above = tile_below = tile_left = tile_right = None

        space = self.get_space(state)
        space_row = space[0]
        space_col = space[1]

        if space_row - 1 >= 0:
            tile_above = state[space_row - 1][space_col]
        if space_row + 1 <= 3:
            tile_below = state[space_row + 1][space_col]
        if space_col - 1 >= 0:
            tile_left = state[space_row][space_col - 1]
        if space_col + 1 <= 3:
            tile_right = state[space_row][space_col + 1]

        if tile_below is not None:
            actions_list.append(('up', tile_below))
        if tile_left is not None:
            actions_list.append(('right', tile_left))
        if tile_above is not None:
            actions_list.append(('down', tile_above))
        if tile_right is not None:
            actions_list.append(('left', tile_right))

        random.shuffle(actions_list)

        return actions_list

    def result(self, state, action):
        new_state = copy.deepcopy(state)
        tile_row = -1
        tile_col = -1

        for row_index, row in enumerate(state):
            if action[1] not in row:
                continue
            tile_row = row_index
            tile_col = row.index(action[1])
            break

        space = self.get_space(state)
        space_row = space[0]
        space_col = space[1]

        if action[0] == 'up':
            new_state[space_row][space_col] = action[1]
            new_state[tile_row][tile_col] = 0
        elif action[0] == 'right':
            new_state[space_row][space_col] = action[1]
            new_state[tile_row][tile_col] = 0
        elif action[0] == 'down':
            new_state[space_row][space_col] = action[1]
            new_state[tile_row][tile_col] = 0
        elif action[0] == 'left':
            new_state[space_row][space_col] = action[1]
            new_state[tile_row][tile_col] = 0

        return new_state

    def solved(self, state):
        return state == self.solved_state

    def get_solution(self, node):
        solution = []

        while node.parent:
            solution.append(node.action)
            node = node.parent

        solution.reverse()
        return solution

    def search(self):
        frontier = PriorityQueueFrontier()
        initial_node = Node(self.initial_state, None, None, 0)
        frontier.add(initial_node)

        explored = []

        while True:
            if frontier.is_empty():
                raise Exception("No Solution")
            else:
                pass
                print(f'{len(frontier.nodes)} nodes in frontier')

            node = frontier.pop()

            if self.solved(node.state):
                solution = self.get_solution(node)
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                print(solution)
                return

            explored.append(node.state)

            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, node, action, node.cost + 1)

                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)

    def get_space(self, state):
        for row_index, row in enumerate(state):
            if 0 not in row:
                continue

            space_col = row.index(0)
            space_row = row_index
            return space_row, space_col

        return None


if __name__ == "__main__":
    initial_state = [[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 14, 10, 12],
                     [0, 13, 11, 15]]

    puzzle = Puzzle(initial_state)
    puzzle.search()
