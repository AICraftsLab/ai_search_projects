import sys

import pygame

pygame.init()


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

    pygame.display.set_caption('Maze')
    screen = pygame.display.set_mode((maze_game.width, maze_game.height))

    solve_font = pygame.font.SysFont("sanscomic", int(maze_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (maze_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (maze_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                maze_game.move('up')
            elif keys[pygame.K_RIGHT]:
                maze_game.move('right')
            elif keys[pygame.K_DOWN]:
                maze_game.move('down')
            elif keys[pygame.K_LEFT]:
                maze_game.move('left')

        #pygame.draw.rect(screen, (0, 0, 0), (0, 0, WIDTH, HEIGHT))
        maze_game.draw(screen)

        if maze_game.solved():
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        maze_filepath = sys.argv[1]
        run(maze_filepath)
    else:
        maze_filepath = 'maze2.txt'
        run(maze_filepath)
