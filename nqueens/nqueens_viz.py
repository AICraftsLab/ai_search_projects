import sys

import pygame
import copy
import random

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


class NQueens:
    def __init__(self, n=5):
        self.n = n

    def actions(self, state):
        queens_pos, attack_cols, attack_pos_diag, attack_neg_diag = state
        queen_row = len(queens_pos)

        actions = []
        for col in range(self.n):
            pos_diag = queen_row + col
            neg_diag = queen_row - col
            if col not in attack_cols and pos_diag not in attack_pos_diag and neg_diag not in attack_neg_diag:
                actions.append((queen_row, col))

        random.shuffle(actions)
        return actions

    def result(self, state, action):
        new_state = copy.deepcopy(state)
        queens_pos, attack_cols, attack_pos_diag, attack_neg_diag = new_state

        queens_pos.append(action)
        attack_cols.append(action[1])
        attack_pos_diag.append(action[0] + action[1])
        attack_neg_diag.append(action[0] - action[1])

        return new_state

    def print_state(self, state):
        queens_pos, *_ = state

        print()
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) in queens_pos:
                    print('Q', end=' ')
                else:
                    print('*', end=' ')
            print()

    def solved(self, state):
        queens_pos, *_ = state

        if len(queens_pos) != self.n:
            return False

        return True


class Cell:
    size = 50
    border_width = 1
    q_color = 'green'
    attacked_q_color = 'red'
    queen_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.__under_attack = False
        self.has_queen = False
        self.x = col * Cell.size
        self.y = row * Cell.size
        self.rect = pygame.Rect(self.x, self.y, Cell.size, Cell.size)

    def is_clicked(self, click_pos):
        return self.rect.collidepoint(click_pos)

    def set_under_attack(self, under_attack):
        if not self.has_queen:
            return

        self.__under_attack = under_attack

    def draw(self, screen):
        pygame.draw.rect(screen, 'gray', (self.x, self.y, Cell.size, Cell.size))
        inner_size = Cell.size - (Cell.border_width * 2)
        pygame.draw.rect(screen, self.color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))

        if self.has_queen:
            color = Cell.q_color if not self.__under_attack else Cell.attacked_q_color
            queen_text = Cell.queen_font.render('Q', 1, color)
            text_x = self.x + (Cell.size / 2) - (queen_text.get_width() / 2)
            text_y = self.y + (Cell.size / 2) - (queen_text.get_height() / 2)
            screen.blit(queen_text, (text_x, text_y))


class NQueensGame:
    def __init__(self, n=8):
        if n < 4:
            raise Exception('n must be greater than 3')
        self.n = n
        self.width = n * Cell.size
        self.height = n * Cell.size

        self.cells = []
        color = 'black'
        for i in range(n):
            row = []
            for j in range(n):
                cell = Cell(i, j, color)
                color = 'black' if color == 'white' else 'white'
                row.append(cell)
            self.cells.append(row)
            if n % 2 == 0:
                color = 'black' if color == 'white' else 'white'

    def draw(self, screen):
        for row in self.cells:
            for cell in row:
                cell.draw(screen)

    def set_state(self, state):
        queens_pos, *_ = state

        for row in self.cells:
            for cell in row:
                if (cell.row, cell.col) in queens_pos:
                    cell.has_queen = True
                else:
                    cell.has_queen = False


def run(n=None):
    if n is None:
        nqueens_game = NQueensGame()
    else:
        nqueens_game = NQueensGame(n)

    nqueens = NQueens(nqueens_game.n)

    frontier = StackFrontier()

    initial_state = ([], [], [], [])
    frontier.add(Node(initial_state, None, None, 0))

    explored = []

    screen = pygame.display.set_mode((nqueens_game.width, nqueens_game.height))
    pygame.display.set_caption('N-Queens')
    clock = pygame.time.Clock()

    fps = 5
    solved = False

    solve_font = pygame.font.SysFont("sanscomic", int(nqueens_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (nqueens_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (nqueens_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        if solved:
            continue

        if len(frontier.nodes) == 0:
            raise Exception('No solution')
        else:
            print(f'{len(frontier.nodes)} in frontier')

        node = frontier.pop()
        explored.append(node.state)
        nqueens_game.set_state(node.state)

        if nqueens.solved(node.state):
            nqueens.print_state(node.state)
            solved = True
            print('Solution found')
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')

        for action in nqueens.actions(node.state):
            new_state = nqueens.result(node.state, action)
            new_node = Node(new_state, action, node, node.cost + 1)

            if not frontier.contains(node) and new_node.state not in explored:
                frontier.add(new_node)

        nqueens_game.draw(screen)

        if solved:
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        pygame.display.flip()
        clock.tick(fps)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run()
    else:
        try:
            n = int(sys.argv[1])
            run(n)
        except ValueError:
            print('Usage: python nqueens_pygame [n]')
