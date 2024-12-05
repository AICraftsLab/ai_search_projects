import random
import pygame

# Initialize pygame
pygame.init()


class Tile:
    # Tiles size, border width, and font
    size = 80
    border_width = 2
    tile_num_font = pygame.font.SysFont("comicsans", int(size / 1.5))

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

        # List to hold all tiles
        self.tiles = []

        # Start with tile number 1
        number = 1

        for i in range(self.size):
            row = []
            for j in range(self.size):
                # Create a new tile
                tile = Tile(number, i, j)

                # The last tile is the empty space
                if number == self.size ** 2:
                    tile = Tile(0, i, j)
                    self.space = tile

                row.append(tile)

                number += 1
            self.tiles.append(row)

    def draw(self, surface):
        # Draw all tiles except the empty space
        for row in self.tiles:
            for tile in row:
                if tile.number != 0:
                    tile.draw(surface)

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

    def solved(self):
        # Check if the puzzle is solved
        solved_puzzle = [[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 0]]

        # Compare each tile with the solved puzzle state
        # Return False if any tile is not in the correct position
        # Return True if all tiles are in the correct position
        for i, row in enumerate(self.tiles):
            for j, tile in enumerate(row):
                if tile.number != solved_puzzle[i][j]:
                    return False
        return True

    def shuffle(self, moves_num=20):
        # Shuffle the puzzle by making random moves
        moves = ['up', 'right', 'down', 'left']

        for i in range(moves_num):
            move = random.choice(moves)
            self.slide(move)


def run():
    puzzle_game = PuzzleGame()
    puzzle_game.shuffle()

    # Set window title and size
    pygame.display.set_caption('15-Puzzle')
    surface = pygame.display.set_mode((puzzle_game.width, puzzle_game.height))

    # Clock to control fps
    clock = pygame.time.Clock()

    # Create the "Solved" text to display when the goal is reached
    solve_font = pygame.font.SysFont("comicsans", int(puzzle_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (puzzle_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (puzzle_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            # Exit if the window is closed
            if event.type == pygame.QUIT:
                quit()
            
            # Get the pressed key and move accordingly
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    puzzle_game.slide('up')
                elif keys[pygame.K_RIGHT]:
                    puzzle_game.slide('right')
                elif keys[pygame.K_DOWN]:
                    puzzle_game.slide('down')
                elif keys[pygame.K_LEFT]:
                    puzzle_game.slide('left')

        # Fill screen with black color
        surface.fill('black')

        # Draw the puzzle
        puzzle_game.draw(surface)

        # Display the "Solved" text if the goal is reached
        if puzzle_game.solved():
            surface.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(60)


# Entry point of the script
if __name__ == '__main__':
    run()
