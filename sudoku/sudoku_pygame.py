import pygame
import sudoku_gen

pygame.init()


class Cell:
    size = 50
    border_width = 1
    changeable_cell_color = 'green'
    nonchangeable_cell_color = 'dimgray'
    invalid_cell_color = 'red'
    cell_num_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

    def __init__(self, number, row, col, changeable=False):
        self.number = number
        self.row = row
        self.col = col
        self.x = col * Cell.size
        self.y = row * Cell.size
        self.rect = pygame.Rect(self.x, self.y, Cell.size, Cell.size)
        self.invalid = False
        self.changeable = changeable

        if changeable:
            self.color = Cell.changeable_cell_color
        else:
            self.color = Cell.nonchangeable_cell_color

    def is_clicked(self, click_pos):
        return self.rect.collidepoint(click_pos)

    def set_invalid(self, invalid):
        if not self.changeable:
            return

        if invalid:
            self.color = Cell.invalid_cell_color
        else:
            self.color = Cell.changeable_cell_color

    def draw(self, screen):
        pygame.draw.rect(screen, 'black', (self.x, self.y, Cell.size, Cell.size))
        inner_size = Cell.size - (Cell.border_width * 2)
        color = 'tan'
        pygame.draw.rect(screen, color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))

        if self.number != 0:
            cell_num_text = Cell.cell_num_font.render(str(self.number), 1, self.color)
            text_x = self.x + (Cell.size / 2) - (cell_num_text.get_width() / 2)
            text_y = self.y + (Cell.size / 2) - (cell_num_text.get_height() / 2)
            screen.blit(cell_num_text, (text_x, text_y))


class SudokuGame:
    def __init__(self):
        sudoku = sudoku_gen.Sudoku()

        self.initial_state = sudoku.initial_state
        self.grid_size = sudoku.grid_size
        self.box_size = sudoku.box_size

        self.width = self.grid_size * Cell.size
        self.height = self.grid_size * Cell.size

        self.cells = []
        for i, nums_row in enumerate(self.initial_state):
            cells_row = []
            for j, num in enumerate(nums_row):
                cell = Cell(num, i, j, num == 0)
                cells_row.append(cell)
            self.cells.append(cells_row)

    def click_cell_at_pos(self, click_pos):
        clicked_cell = None
        for row in self.cells:
            for cell in row:
                if cell.is_clicked(click_pos):
                    self.change_cell_number(cell)
                    clicked_cell = cell
                    break
            if clicked_cell is not None:
                break

    def change_cell_number(self, cell):

        number = cell.number

        number += 1
        number %= self.grid_size
        if number == 0:
            number = self.grid_size

        cell.number = number

    def get_grid(self):
        grid = []
        for cells_row in self.cells:
            nums_row = []
            for cell in cells_row:
                nums_row.append(cell.number)
            grid.append(nums_row)

        return grid

    def check_row_validity(self, state, number, position):
        for i, row in enumerate(state):
            if i == position[0] and row.count(number) == 1:
                return True

        return False

    def check_col_validity(self, state, number, position):
        col = [row[position[1]] for row in state]

        return col.count(number) == 1

    def check_box_validity(self, state, number, position):
        box_row = int(position[0] / self.box_size)
        box_col = int(position[1] / self.box_size)
        box_numbers = []

        for i, row in enumerate(state):
            if int(i / self.box_size) == box_row:
                for j, num in enumerate(row):
                    if int(j / self.box_size) == box_col:
                        box_numbers.append(num)

        if box_numbers.count(number) == 1:
            return True

        return False

    def check_validity(self, state, number, position):
        num_row_validity = self.check_row_validity(state, number, position)
        num_col_validity = self.check_col_validity(state, number, position)
        num_box_validity = self.check_box_validity(state, number, position)

        return all([num_row_validity, num_col_validity, num_box_validity])

    def solved(self, state):
        solved = True

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                no_zero = state[i].count(0) == 0
                valid = self.check_validity(state, state[i][j], (i, j))

                if not valid:
                    self.cells[i][j].set_invalid(True)
                else:
                    self.cells[i][j].set_invalid(False)

                if not no_zero or not valid:
                    solved = False

        return solved

    def draw(self, screen):
        for row in self.cells:
            for cell in row:
                cell.draw(screen)


def run():
    sudoku_game = SudokuGame()

    screen = pygame.display.set_mode((sudoku_game.width, sudoku_game.height))
    pygame.display.set_caption('Sudoku')

    solve_font = pygame.font.SysFont("sanscomic", int(sudoku_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (sudoku_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (sudoku_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                sudoku_game.click_cell_at_pos(event.pos)

        sudoku_game.draw(screen)

        if sudoku_game.solved(sudoku_game.get_grid()):
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()


if __name__ == '__main__':
    run()
