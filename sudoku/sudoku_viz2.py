import copy
import math
import random

import pygame
from sudoku_gen import SudokuGenerator

pygame.init()


class Node:
    def __init__(self, state, action, parent, cost):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost


class StackFrontier:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def contains(self, node):
        for n in self.nodes:
            if n.state == node.state:
                return True

        return False

    def is_empty(self):
        return len(self.nodes) == 0

    def pop(self):
        return self.nodes.pop()


class QueueFrontier(StackFrontier):
    def pop(self):
        return self.nodes.pop(0)


class Sudoku:
    def __init__(self, initial_state=None):
        self.initial_state = initial_state

        rows = len(self.initial_state)
        cols = len(self.initial_state[0])

        if rows != cols:
            raise Exception("Invalid grid size: must be a perfect square")

        if not all([len(row) == cols for row in self.initial_state]):
            raise Exception("Invalid grid size: all rows must be equal in size")

        self.grid_size = rows
        self.box_size = int(math.isqrt(self.grid_size))

    def actions(self, state):
        next_space = None
        actions = []

        for i, row in enumerate(state):
            for j, num in enumerate(row):
                if num == 0:
                    next_space = (i, j)
                    break
            if next_space is not None:
                break

        if next_space is None:
            return actions

        for i in range(1, self.grid_size + 1):
            if self.check_validity(state, i, next_space):
                actions.append((next_space, i))

        random.shuffle(actions)
        return actions

    def check_row_validity(self, state, number, position, solved=False):
        for i, row in enumerate(state):
            if not solved:
                if i == position[0] and row.count(number) == 0:
                    return True
            else:
                if i == position[0] and row.count(number) == 1:
                    return True

        return False

    def check_col_validity(self, state, number, position, solved=False):
        col = [row[position[1]] for row in state]

        if not solved:
            return col.count(number) == 0
        else:
            return col.count(number) == 1

    def check_box_validity(self, state, number, position, solved=False):
        box_row = int(position[0] / self.box_size)
        box_col = int(position[1] / self.box_size)
        box_numbers = []

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

    def check_validity(self, state, number, position, solved=False):
        num_row_validity = self.check_row_validity(state, number, position, solved)
        num_col_validity = self.check_col_validity(state, number, position, solved)
        num_box_validity = self.check_box_validity(state, number, position, solved)

        return all([num_row_validity, num_col_validity, num_box_validity])

    def result(self, state, action):
        new_state = copy.deepcopy(state)
        row, col = action[0][0], action[0][1]

        new_state[row][col] = action[1]
        return new_state

    def solved(self, state):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                no_zero = state[i].count(0) == 0
                if not no_zero or not self.check_validity(state, state[i][j], (i, j), solved=True):
                    return False

        return True

    def print_state(self, state):
        print()
        for row in state:
            print(row)
        print()


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
        sudoku = SudokuGenerator('hard')

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

    # def click_cell_at_pos(self, click_pos):
    #     clicked_cell = None
    #     for row in self.cells:
    #         for cell in row:
    #             if cell.is_clicked(click_pos):
    #                 self.change_cell_number(cell)
    #                 clicked_cell = cell
    #                 break
    #         if clicked_cell is not None:
    #             break

    # def change_cell_number(self, cell, number):
    #     cell.number = number

    def set_state(self, state):
        for i, row in enumerate(state):
            for j, num in enumerate(row):
                cell = self.cells[i][j]
                if cell.changeable:
                    cell.number = num

    # '''
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

    # '''

    def draw(self, screen):
        for row in self.cells:
            for cell in row:
                cell.draw(screen)


def run():
    sudoku_game = SudokuGame()

    screen = pygame.display.set_mode((sudoku_game.width, sudoku_game.height))
    pygame.display.set_caption('Sudoku Solver')
    clock = pygame.time.Clock()

    sudoku = Sudoku(sudoku_game.initial_state)
    frontier = StackFrontier()
    frontier.add(Node(sudoku_game.initial_state, None, None, 0))

    explored = []

    solve_font = pygame.font.SysFont("sanscomic", int(sudoku_game.width / 6))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (sudoku_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (sudoku_game.height / 2) - (solve_text.get_height() / 2)

    solved = False
    fps = 2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        # *****************************
        if solved:
            continue

        if frontier.is_empty():
            raise Exception("No solution")
        else:
            print(f'{len(frontier.nodes)} nodes in frontier')

        node = frontier.pop()
        sudoku_game.set_state(node.state)

        if sudoku.solved(node.state):
            sudoku.print_state(sudoku.initial_state)
            sudoku.print_state(node.state)
            solved = True
            print('Solution found')
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')

        explored.append(node.state)

        for action in sudoku.actions(node.state):
            new_state = sudoku.result(node.state, action)
            new_node = Node(new_state, action, node, node.cost + 1)

            if not frontier.contains(new_node) and new_node.state not in explored:
                frontier.add(new_node)

        sudoku_game.draw(screen)

        # if solved:
        if sudoku_game.solved(sudoku_game.get_grid()):
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()
        clock.tick(fps)


if __name__ == '__main__':
    run()
