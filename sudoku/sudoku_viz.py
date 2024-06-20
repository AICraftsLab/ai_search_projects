import copy
import random
import pygame
from node import Node
from frontier import StackFrontier
from sudoku_gen import SudokuGenerator

pygame.init()


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

class Cell:
    # Class level details
    size = 50
    border_width = 1
    changeable_cell_color = 'green'
    nonchangeable_cell_color = 'dimgray'
    invalid_cell_color = 'red'
    cell_num_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

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
        sudoku_gen = SudokuGenerator('hard')

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
                # Changeable if num == 0, esle not
                cell = Cell(num, i, j, num == 0)
                cells_row.append(cell)
            self.cells.append(cells_row)

    def set_state(self, state):
        # Set the state of the Sudoku grid to a given state
        for i, row in enumerate(state):
            for j, num in enumerate(row):
                cell = self.cells[i][j]
                if cell.changeable:
                    cell.number = num

    def draw(self, surface):
        # Draw the Sudoku grid
        for row in self.cells:
            for cell in row:
                cell.draw(surface)


def run():
    # Initialize the Sudoku game
    sudoku_game = SudokuGame()

    # Setup Pygame display
    surface = pygame.display.set_mode((sudoku_game.width, sudoku_game.height))
    pygame.display.set_caption('Sudoku Visualization')
    clock = pygame.time.Clock()
    fps = 5

    # Initialize the Sudoku puzzle solver
    sudoku = Sudoku(sudoku_game.initial_state)
    frontier = StackFrontier()
    frontier.add(Node(sudoku_game.initial_state, None, None, 0))

    explored = []
    solved = False

    # Create font for "Solved" text
    solve_font = pygame.font.SysFont("sanscomic", int(sudoku_game.width / 6))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (sudoku_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (sudoku_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        # If solved, skip to the next iteration
        if solved:
            continue

        # If nothing left in frontier, no solution exists
        if frontier.is_empty():
            raise Exception("No solution")
        else:
            print(f'{len(frontier.nodes)} nodes in frontier')

        # Get a node from the frontier
        node = frontier.pop()

        # Set the game state to the node's state
        sudoku_game.set_state(node.state)

        # If the node is the goal, print the solution
        if sudoku.solved(node.state):
            sudoku.print_state(node.state)
            solved = True
            print('Solution found')
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')

        # Add node to list of explored nodes
        explored.append(node.state)

        # Expand the node
        for action in sudoku.actions(node.state):
            new_state = sudoku.result(node.state, action)
            new_node = Node(new_state, action, node, node.cost + 1)

            if not frontier.contains(new_node) and new_node.state not in explored:
                frontier.add(new_node)

        # Draw the current state of the game
        sudoku_game.draw(surface)

        # Display "Solved" text if the goal is reached
        if solved:
            surface.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(fps)


# Entry point of the script
if __name__ == '__main__':
    run()
