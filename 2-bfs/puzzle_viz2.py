import copy
import random
import pygame
from node import Node
from frontier import QueueFrontier

# Initialize pygame
pygame.init()


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
                if num not in unique_numbers:
                    unique_numbers.append(num)
                else:
                    raise Exception("Duplicate number in puzzle")

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


class Tile:
    # Tiles size, border width, and font
    size = 80
    border_width = 2
    tile_num_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

    def __init__(self, number, row, col):
        # Initialize the tile with a number and its position (row and col)
        self.number = number
        self.row = row
        self.col = col
        self.x = col * Tile.size  # X position
        self.y = row * Tile.size  # Y position

        # Render the number text
        self.tile_num_text = self.tile_num_font.render(str(self.number), 1, 'white')

    def draw(self, surface):
        # Draw the tile (border)
        pygame.draw.rect(surface, 'black', (self.x, self.y, Tile.size, Tile.size))

        # Draw the tile (body)
        inner_size = Tile.size - (Tile.border_width * 2)
        pygame.draw.rect(surface, 'brown',
                         (self.x + Tile.border_width, self.y + Tile.border_width, inner_size, inner_size))

        # Draw the tile (number)
        text_x = self.x + (Tile.size / 2) - (self.tile_num_text.get_width() / 2)
        text_y = self.y + (Tile.size / 2) - (self.tile_num_text.get_height() / 2)
        surface.blit(self.tile_num_text, (text_x, text_y))

    def update(self, row, col):
        # Update the tile position
        self.row = row
        self.col = col
        self.x = col * Tile.size
        self.y = row * Tile.size


class PuzzleGame:
    def __init__(self):
        self.size = 4  # Puzzle size (4x4)

        # Width and height of the puzzle
        self.height = Tile.size * self.size
        self.width = Tile.size * self.size
        self.space = None
        self.tiles = []

        # Fill the board with tiles
        self.populate()

    def populate(self):
        # Create the board's tiles
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

    def draw(self, screen):
        # Draw all tiles except the empty space
        for row in self.tiles:
            for tile in row:
                if tile.number != 0:
                    tile.draw(screen)

    def get_available_tiles(self):
        # Get tiles that can be moved into the empty space
        available_tiles = {}
        tile_above = tile_below = tile_left = tile_right = None

        # Check for available tiles around of the empty space
        if self.space.row - 1 >= 0:
            tile_above = self.tiles[self.space.row - 1][self.space.col]
        if self.space.row + 1 <= self.size - 1:
            tile_below = self.tiles[self.space.row + 1][self.space.col]
        if self.space.col - 1 >= 0:
            tile_left = self.tiles[self.space.row][self.space.col - 1]
        if self.space.col + 1 <= self.size - 1:
            tile_right = self.tiles[self.space.row][self.space.col + 1]

        # Add available tiles to the dictionary with their respective move direction
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
        # Slide a tile into the empty space
        available_tiles = self.get_available_tiles()

        # Get the tile to move based on direction
        tile = available_tiles.get(direction)
        if tile is None:
            return

        tile_row = tile.row
        tile_col = tile.col

        space_row = self.space.row
        space_col = self.space.col

        # Swap the positions of the empty space and the tile
        self.tiles[tile_row][tile_col] = self.tiles[self.space.row][self.space.col]
        self.tiles[self.space.row][self.space.col] = tile

        # Update their positions
        tile.update(space_row, space_col)
        self.space.update(tile_row, tile_col)

    def shuffle(self, moves_num=20):
        # Shuffle the puzzle by making random moves
        moves = ['up', 'right', 'down', 'left']

        for i in range(moves_num):
            move = random.choice(moves)
            self.slide(move)

    def get_state(self):
        # Return a 2D list representation of the game
        state = []

        for tiles_row in self.tiles:
            num_row = []
            for tile in tiles_row:
                num_row.append(tile.number)

            state.append(num_row)

        return state

    def set_state(self, state):
        # Updates the positions of the tiles to match a new state.

        # Get the current tiles into a single list
        tiles = []
        for row in self.tiles:
            for tile in row:
                tiles.append(tile)

        # Clear the current tiles of the board
        self.tiles.clear()

        # Reassign tiles to match the new state
        # and update their positions
        for i, row in enumerate(state):
            new_row = []
            for j, num in enumerate(row):
                for tile in tiles:
                    if tile.number == num:
                        # Update the tile's position
                        tile.update(i, j)
                        new_row.append(tile)
                        break

            # Append the new row to the board's tiles
            self.tiles.append(new_row)


def run():
    # Run the visualization

    # Initialize the puzzle game
    puzzle_game = PuzzleGame()
    puzzle_game.shuffle()

    # Get the shuffled state
    initial_state = puzzle_game.get_state()

    # Initialize the puzzle solver with the initial state
    puzzle = Puzzle(initial_state)

    frontier = QueueFrontier()
    initial_node = Node(initial_state, None, None, 0)
    frontier.add(initial_node)

    explored = []
    solved = False

    # Search/frame rate
    fps = 10

    # Set up Pygame display properties
    pygame.display.set_caption('15-Puzzle Solver')
    screen = pygame.display.set_mode((puzzle_game.width, puzzle_game.height))
    clock = pygame.time.Clock()

    # Set up the 'Solved' text display
    solve_font = pygame.font.SysFont("sanscomic", int(puzzle_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (puzzle_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (puzzle_game.height / 2) - (solve_text.get_height() / 2)

    # Main game loop
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

        # Pop a node from the frontier
        node = frontier.pop()

        # Set the game state to the node's state
        puzzle_game.set_state(node.state)

        # Check if the current state is the solved state
        if puzzle.solved(node.state):
            print(puzzle.get_solution(node))
            print("Solution Found")
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')
            solved = True

        # Add the current state to the explored list
        explored.append(node.state)

        for action in puzzle.actions(node.state):
            new_state = puzzle.result(node.state, action)
            new_node = Node(new_state, node, action, node.cost + 1)

            if not frontier.contains(new_node) and new_node.state not in explored:
                frontier.add(new_node)

        # Clear the screen
        screen.fill('black')

        # Draw the new puzzle state
        puzzle_game.draw(screen)

        # Display 'Solved' text if solution has been found
        if solved:
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(fps)


# Entry point of the script
if __name__ == '__main__':
    run()
