import pygame
from sudoku_gen import SudokuGenerator

pygame.init()


class Cell:
    # Class level details
    size = 50
    border_width = 1
    changeable_cell_color = 'green'
    nonchangeable_cell_color = 'dimgray'
    invalid_cell_color = 'red'
    cell_num_font = pygame.font.SysFont("comicsans", int(size / 1.5))

    def __init__(self, number, row, col, changeable=False):
        #  Initializes a Cell instance
        self.number = number
        self.row = row
        self.col = col
        self.x = col * Cell.size  # X position
        self.y = row * Cell.size  # Y position

        # Rect object for the cell
        self.rect = pygame.Rect(self.x, self.y, Cell.size, Cell.size)

        # Flag to check if the cell is invalid
        self.invalid = False

        # Flag to check if the cell can be changed
        self.changeable = changeable

        if changeable:
            self.color = Cell.changeable_cell_color
        else:
            self.color = Cell.nonchangeable_cell_color

    def is_clicked(self, click_pos):
        # Check if the cell is clicked based on the click position
        # If cell is non-changeable, return False
        if not self.changeable:
            return False
        
        return self.rect.collidepoint(click_pos)

    def set_invalid(self, invalid):
        # Set the cell text color based on its validity

        # If cell is non-changeable, return
        if not self.changeable:
            return

        if invalid:
            self.color = Cell.invalid_cell_color
        else:
            self.color = Cell.changeable_cell_color

    def draw(self, surface):
        # Draw the cell (border)
        pygame.draw.rect(surface, 'black', (self.x, self.y, Cell.size, Cell.size))

        # Draw the cell (body)
        inner_size = Cell.size - (Cell.border_width * 2)
        color = 'tan'  # Cell body color
        pygame.draw.rect(surface, color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))

        # Draw the cell (text)
        if self.number != 0:
            cell_num_text = Cell.cell_num_font.render(str(self.number), 1, self.color)
            text_x = self.x + (Cell.size / 2) - (cell_num_text.get_width() / 2)
            text_y = self.y + (Cell.size / 2) - (cell_num_text.get_height() / 2)
            surface.blit(cell_num_text, (text_x, text_y))


class SudokuGame:
    def __init__(self):
        # Initializes the puzzle

        # Generate a Sudoku puzzle
        sudoku_gen = SudokuGenerator()

        # Get the generated grid
        self.initial_state = sudoku_gen.initial_state

        self.grid_size = sudoku_gen.grid_size
        self.box_size = sudoku_gen.box_size

        # Width and height of the game screen
        self.width = self.grid_size * Cell.size
        self.height = self.grid_size * Cell.size

        self.cells = []

        # Create Cell objects for each position in the grid
        for i, nums_row in enumerate(self.initial_state):
            cells_row = []
            for j, num in enumerate(nums_row):
                # Changeable if num == 0, else not
                cell = Cell(num, i, j, num == 0)
                cells_row.append(cell)
            self.cells.append(cells_row)

    def click_cell_at_pos(self, click_pos):
        # Check which cell is clicked and change its number
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
        # Increment the cell number cyclically
        number = cell.number

        number += 1
        number %= self.grid_size
        if number == 0:
            number = self.grid_size

        cell.number = number

    def get_state(self):
        # Get the current state of the Sudoku grid
        grid = []
        for cells_row in self.cells:
            nums_row = []
            for cell in cells_row:
                nums_row.append(cell.number)
            grid.append(nums_row)

        return grid

    def solved(self):
        # Check if the Sudoku puzzle is solved. Simultaneously
        # change the text color of cells based on validity.
        state = self.get_state()
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

    def check_validity(self, state, number, position):
        # Check if the number is valid in its position
        num_row_validity = self.check_row_validity(state, number, position)
        num_col_validity = self.check_col_validity(state, number, position)
        num_box_validity = self.check_box_validity(state, number, position)

        return all([num_row_validity, num_col_validity, num_box_validity])

    def check_row_validity(self, state, number, position):
        # Check if the number is valid in its row
        for i, row in enumerate(state):
            if i == position[0] and row.count(number) == 1:
                return True

        return False

    def check_col_validity(self, state, number, position):
        # Check if the number is valid in its column
        col = [row[position[1]] for row in state]

        return col.count(number) == 1

    def check_box_validity(self, state, number, position):
        # Check if the number is valid in its sub-grid
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

    def draw(self, surface):
        # Draw the Sudoku grid
        for row in self.cells:
            for cell in row:
                cell.draw(surface)


def run():
    # Run the game logic
    sudoku_game = SudokuGame()  # Create a Sudoku game instance

    # Set the display
    surface = pygame.display.set_mode((sudoku_game.width, sudoku_game.height))
    pygame.display.set_caption('Sudoku')

    # Clock to control fps
    clock = pygame.time.Clock()

    # Create the "Solved" text to display when the goal is reached
    solve_font = pygame.font.SysFont("comicsans", int(sudoku_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (sudoku_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (sudoku_game.height / 2) - (solve_text.get_height() / 2)

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            # Get the click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                sudoku_game.click_cell_at_pos(event.pos)

        # Draw the puzzle
        sudoku_game.draw(surface)

        # Display the "Solved" text if the goal is reached
        if sudoku_game.solved():
            surface.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(60)


# Entry point of the script
if __name__ == '__main__':
    run()
