import random

import pygame

pygame.init()


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

    def solved(self):
        solved_puzzle = [[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 0]]

        for i, row in enumerate(self.tiles):
            for j, tile in enumerate(row):
                if not tile.number == solved_puzzle[i][j]:
                    return False
        return True

    def shuffle(self, moves_num=20):
        moves = ['up', 'right', 'down', 'left']

        for i in range(moves_num):
            move = random.choice(moves)
            self.slide(move)


def run():
    puzzle_game = PuzzleGame()
    puzzle_game.shuffle()

    pygame.display.set_caption('15-Puzzle')
    screen = pygame.display.set_mode((puzzle_game.width, puzzle_game.height))

    solve_font = pygame.font.SysFont("sanscomic", int(puzzle_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (puzzle_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (puzzle_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                puzzle_game.slide('up')
            elif keys[pygame.K_RIGHT]:
                puzzle_game.slide('right')
            elif keys[pygame.K_DOWN]:
                puzzle_game.slide('down')
            elif keys[pygame.K_LEFT]:
                puzzle_game.slide('left')

        pygame.draw.rect(screen, (0, 0, 0), (0, 0, puzzle_game.width, puzzle_game.height))
        puzzle_game.draw(screen)

        if puzzle_game.solved():
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()


if __name__ == '__main__':
    run()
