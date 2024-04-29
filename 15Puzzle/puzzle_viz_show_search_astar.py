import copy
import heapq
import random

import pygame

pygame.init()


class Node:
    as_astar = False

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
        if Node.as_astar:
            return self.manhattan_distance() + self.cost < other.manhattan_distance() + self.cost
        else:
            return self.manhattan_distance() < other.manhattan_distance()


class StackFrontier:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def pop(self):
        return self.nodes.pop()

    def contains(self, node):
        for n in self.nodes:
            if n.state == node.state:
                return True

        return False

    def is_empty(self):
        return len(self.nodes) == 0


class QueueFrontier(StackFrontier):
    def pop(self):
        return self.nodes.pop(0)


class PriorityQueueFrontier(StackFrontier):
    def add(self, node):
        heapq.heappush(self.nodes, node)

    def pop(self):
        return heapq.heappop(self.nodes)


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
        frontier = QueueFrontier()
        initial_node = Node(self.initial_state, None, None, 0)
        frontier.add(initial_node)

        explored = []

        while True:
            if frontier.is_empty():
                print("No Solution")
                return
            else:
                pass
                print(f'{len(frontier.nodes)} nodes in frontier')

            node = frontier.pop()

            if self.solved(node.state):
                solution = self.get_solution(node)
                print("Solution Found")
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


class Tile:
    size = 80
    border_width = 2
    __tile_num_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

    def __init__(self, number, row, col):
        self.number = number
        self.row = row
        self.col = col
        self.x = col * Tile.size
        self.y = row * Tile.size
        self.__tile_num_text = self.__tile_num_font.render(str(self.number), 1, 'white')

    def draw(self, screen):
        pygame.draw.rect(screen, 'black', (self.x, self.y, Tile.size, Tile.size))
        inner_size = Tile.size - (Tile.border_width * 2)
        pygame.draw.rect(screen, 'brown',
                         (self.x + Tile.border_width, self.y + Tile.border_width, inner_size, inner_size))

        text_x = self.x + (Tile.size / 2) - (self.__tile_num_text.get_width() / 2)
        text_y = self.y + (Tile.size / 2) - (self.__tile_num_text.get_height() / 2)
        screen.blit(self.__tile_num_text, (text_x, text_y))

    def update(self, row, col):
        self.row = row
        self.col = col
        self.x = col * Tile.size
        self.y = row * Tile.size


class PuzzleGame:
    def __init__(self):
        self.size = 4
        self.height = Tile.size * self.size
        self.width = Tile.size * self.size
        self.tiles = []

        self.populate()

    def draw(self, screen):
        for row in self.tiles:
            for tile in row:
                if tile.number != 0:
                    tile.draw(screen)

    def get_available_tiles(self):
        available_tiles = {}
        tile_above = tile_below = tile_left = tile_right = None

        if self.space.row - 1 >= 0:
            tile_above = self.tiles[self.space.row - 1][self.space.col]
        if self.space.row + 1 <= self.size - 1:
            tile_below = self.tiles[self.space.row + 1][self.space.col]
        if self.space.col - 1 >= 0:
            tile_left = self.tiles[self.space.row][self.space.col - 1]
        if self.space.col + 1 <= self.size - 1:
            tile_right = self.tiles[self.space.row][self.space.col + 1]

        if tile_below is not None:
            available_tiles['up'] = tile_below
        if tile_left is not None:
            available_tiles['right'] = tile_left
        if tile_above is not None:
            available_tiles['down'] = tile_above
        if tile_right is not None:
            available_tiles['left'] = tile_right

        return available_tiles

    def slide(self, direction):
        available_tiles = self.get_available_tiles()

        tile = available_tiles.get(direction)
        if tile is None:
            return

        tile_row = tile.row
        tile_col = tile.col

        space_row = self.space.row
        space_col = self.space.col

        self.tiles[tile_row][tile_col] = self.tiles[self.space.row][self.space.col]
        self.tiles[self.space.row][self.space.col] = tile

        tile.update(space_row, space_col)
        self.space.update(tile_row, tile_col)

    def get_state(self):
        state = []

        for tiles_row in self.tiles:
            num_row = []
            for tile in tiles_row:
                num_row.append(tile.number)

            state.append(num_row)

        return state

    def set_state(self, state):
        tiles = []
        for row in self.tiles:
            for tile in row:
                tiles.append(tile)

        self.tiles.clear()

        for i, row in enumerate(state):
            new_row = []
            for j, num in enumerate(row):
                for tile in tiles:
                    if tile.number == num:
                        tile.update(i, j)
                        new_row.append(tile)
                        break
            self.tiles.append(new_row)

    def populate(self):
        number = 1
        for i in range(self.size):
            row = []
            for j in range(self.size):
                tile = Tile(number, i, j)
                if number == self.size ** 2:
                    tile = Tile(0, i, j)
                    self.space = tile

                row.append(tile)

                number += 1
            self.tiles.append(row)

    def shuffle(self, moves_num=20):
        moves = ['up', 'right', 'down', 'left']

        for i in range(moves_num):
            move = random.choice(moves)
            self.slide(move)


def run():
    puzzle_game = PuzzleGame()
    puzzle_game.shuffle(30)
    initial_state = puzzle_game.get_state()

    puzzle = Puzzle(initial_state)

    frontier = PriorityQueueFrontier()
    Node.as_astar = True
    initial_node = Node(initial_state, None, None, 0)
    frontier.add(initial_node)

    explored = []
    solved = False

    fps = 5

    pygame.display.set_caption('15-Puzzle Solver')
    screen = pygame.display.set_mode((puzzle_game.width, puzzle_game.height))
    clock = pygame.time.Clock()

    solve_font = pygame.font.SysFont("sanscomic", int(puzzle_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (puzzle_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (puzzle_game.height / 2) - (solve_text.get_height() / 2)

    puzzle_game.draw(screen)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        if solved:
            continue

        if frontier.is_empty():
            raise Exception("No Solution")
        else:
            print(f'{len(frontier.nodes)} nodes in frontier')

        node = frontier.pop()
        puzzle_game.set_state(node.state)

        if puzzle.solved(node.state):
            print("Solution Found")
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')
            print(puzzle.get_solution(node))
            solved = True

        explored.append(node.state)

        for action in puzzle.actions(node.state):
            new_state = puzzle.result(node.state, action)
            new_node = Node(new_state, node, action, node.cost + 1)

            if not frontier.contains(new_node) and new_node.state not in explored:
                frontier.add(new_node)

        screen.fill('black')
        puzzle_game.draw(screen)

        if solved:
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()
        clock.tick(fps)


if __name__ == '__main__':
    run()
