import sys
import pygame
import copy
import random
from node import Node
from frontier import StackFrontier

pygame.init()


class NQueens:
    def __init__(self, n=5):
        # Initialize the N-Queens problem with a board of size n
        if n < 4:
            raise Exception('n must be greater than 3')

        self.n = n

    def actions(self, state):
        # Get the possible actions (positions) for placing the next queen
        queens_pos, attack_cols, attack_pos_diag, attack_neg_diag = state

        # Row to place the queen
        queen_row = len(queens_pos)

        actions = []

        # Column to place the queen
        for col in range(self.n):
            # Diagonals of the queen
            pos_diag = queen_row + col
            neg_diag = queen_row - col
            if col not in attack_cols and pos_diag not in attack_pos_diag and neg_diag not in attack_neg_diag:
                actions.append((queen_row, col))

        # Shuffle actions to introduce randomness
        random.shuffle(actions)
        return actions

    def result(self, state, action):
        # Return the new state after performing the given action (placing a queen)
        new_state = copy.deepcopy(state)
        queens_pos, attack_cols, attack_pos_diag, attack_neg_diag = new_state

        queens_pos.append(action)
        attack_cols.append(action[1])
        attack_pos_diag.append(action[0] + action[1])
        attack_neg_diag.append(action[0] - action[1])

        return new_state

    def print_state(self, state):
        # Print the board with queens positions marked
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
        # Check if the state is a solution (all queens are placed)
        queens_pos, *_ = state

        if len(queens_pos) != self.n:
            return False

        return True


class Cell:
    # Class level details
    size = 50
    border_width = 1
    q_color = 'green'  # Color of a queen
    attacked_q_color = 'red'  # Color of an attacked queen
    queen_font = pygame.font.SysFont("comicsans", int(size / 1.5))

    def __init__(self, row, col, color):
        # Initializes a Cell instance
        self.row = row
        self.col = col
        self.color = color  # Cell's color (black or white)
        self.under_attack = False  # Under attack or not
        self.has_queen = False  # Cell has queen or not
        self.x = col * Cell.size  # X position
        self.y = row * Cell.size  # Y position

        # Rect object for the cell
        self.rect = pygame.Rect(self.x, self.y, Cell.size, Cell.size)

    def is_clicked(self, click_pos):
        # Check if the cell is clicked based on the click position
        return self.rect.collidepoint(click_pos)

    def set_under_attack(self, under_attack):
        # Mark the cell as under attack
        if not self.has_queen:
            return

        self.under_attack = under_attack

    def draw(self, surface):
        # Draw the cell (border)
        pygame.draw.rect(surface, 'gray', (self.x, self.y, Cell.size, Cell.size))

        # Draw the cell (body)
        inner_size = Cell.size - (Cell.border_width * 2)
        pygame.draw.rect(surface, self.color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))

        # Draw the cell (text)
        if self.has_queen:
            color = Cell.q_color if not self.under_attack else Cell.attacked_q_color
            queen_text = Cell.queen_font.render('Q', 1, color)
            text_x = self.x + (Cell.size / 2) - (queen_text.get_width() / 2)
            text_y = self.y + (Cell.size / 2) - (queen_text.get_height() / 2)
            surface.blit(queen_text, (text_x, text_y))


class NQueensGame:
    def __init__(self, n):
        # Initializes the puzzle
        if n < 4:
            raise Exception('n must be greater than 3')
        self.n = n

        # Width and height of the game screen
        self.width = n * Cell.size
        self.height = n * Cell.size

        self.cells = []
        self.populate(n)

    def populate(self, n):
        # Populates the board with nxn cells

        # Initial color for the checkerboard pattern
        color = 'black'

        for i in range(n):
            row = []
            for j in range(n):
                # Create each cell with alternating colors
                cell = Cell(i, j, color)
                color = 'black' if color == 'white' else 'white'
                row.append(cell)
            self.cells.append(row)
            if n % 2 == 0:
                color = 'black' if color == 'white' else 'white'

    def draw(self, surface):
        # Draw the game board
        for row in self.cells:
            for cell in row:
                cell.draw(surface)

    def set_state(self, state):
        # Updates the board to match the given state
        queens_pos, *_ = state

        for row in self.cells:
            for cell in row:
                # checks if the current cell's position
                # is in the list of queen positions
                if (cell.row, cell.col) in queens_pos:
                    cell.has_queen = True
                else:
                    cell.has_queen = False


def run(n=5):
    # Run the visualization
    nqueens_game = NQueensGame(n)
    nqueens = NQueens(nqueens_game.n)
    frontier = StackFrontier()

    # Initial state with no queens placed
    initial_state = ([], [], [], [])
    frontier.add(Node(initial_state, None, None, 0))

    explored = []

    # Set the display
    screen = pygame.display.set_mode((nqueens_game.width, nqueens_game.height))
    pygame.display.set_caption('N-Queens')
    clock = pygame.time.Clock()

    fps = 5
    solved = False

    # Create the "Solved" text to display when the goal is reached
    solve_font = pygame.font.SysFont("comicsans", int(nqueens_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (nqueens_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (nqueens_game.height / 2) - (solve_text.get_height() / 2)

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        if solved:
            continue

        # If nothing is left in frontier, no solution exists
        if len(frontier.nodes) == 0:
            raise Exception('No solution')
        else:
            print(f'{len(frontier.nodes)} in frontier')

        # Pop a node from the frontier
        node = frontier.pop()

        # Updates the game state with the current node's state
        nqueens_game.set_state(node.state)

        if nqueens.solved(node.state):
            nqueens.print_state(node.state)
            solved = True
            print('Solution found')
            print(f'Cost:{node.cost}')
            print(f'Nodes explored:{len(explored)}')

        # Add the current state to the explored list
        explored.append(node.state)

        # Expand the node
        for action in nqueens.actions(node.state):
            new_state = nqueens.result(node.state, action)
            new_node = Node(new_state, action, node, node.cost + 1)

            # Check if the node is already in the frontier or explored
            if not frontier.contains(node) and new_node.state not in explored:
                frontier.add(new_node)

        # Draw the board
        nqueens_game.draw(screen)

        # Display the "Solved" text if the goal is reached
        if solved:
            screen.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(fps)


# Entry point of the script
if __name__ == '__main__':
    if len(sys.argv) == 1:
        run()
    else:
        try:
            n = int(sys.argv[1])
            run(n)
        except ValueError:
            print('Usage: python nqueens_pygame [n]')
