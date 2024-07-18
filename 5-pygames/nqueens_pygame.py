import sys
import pygame

pygame.init()


class Cell:
    # Class level details
    size = 50
    border_width = 1
    q_color = 'green'  # Color of a queen
    attacked_q_color = 'red'  # Color of an attacked queen
    queen_font = pygame.font.SysFont("sanscomic", int(size / 1.5))

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

    def click_cell_at_pos(self, click_pos):
        # Handle click event at the given position
        clicked_cell = None
        for row in self.cells:
            for cell in row:
                if cell.is_clicked(click_pos):
                    # Toggle the queen's presence on the clicked cell
                    # making sure number of queens don't exceed n
                    if len(self.get_queens_pos()) < self.n:
                        cell.has_queen = not cell.has_queen
                    else:
                        cell.has_queen = False
                    clicked_cell = cell
                    break
            # Break loop is click event is handled
            if clicked_cell is not None:
                break

    def get_queens_pos(self):
        # Get the positions of all queens on the board
        queens_pos = []
        for row in self.cells:
            for cell in row:
                if cell.has_queen:
                    queens_pos.append((cell.row, cell.col))

        return queens_pos

    def draw(self, surface):
        # Draw the game board
        for row in self.cells:
            for cell in row:
                cell.draw(surface)

    def solved(self):
        # Check if the game is solved (no queens are attacking each other)
        solved = True
        queens_pos = self.get_queens_pos()

        if len(queens_pos) != self.n:
            solved = False

        attack_rows = []
        attack_cols = []
        attack_pos_diag = []
        attack_neg_diag = []

        # Getting attacked cells
        for row, col in queens_pos:
            attack_rows.append(row)
            attack_cols.append(col)
            attack_pos_diag.append(row + col)
            attack_neg_diag.append(row - col)

        for row, col in queens_pos:
            # Current queen's diagonals
            pos_diag = row + col
            neg_diag = row - col

            # Check if the current queen is safe
            cond1 = attack_rows.count(row) == 1
            cond2 = attack_cols.count(col) == 1
            cond3 = attack_pos_diag.count(pos_diag) == 1
            cond4 = attack_neg_diag.count(neg_diag) == 1
            if all([cond1, cond2, cond3, cond4]):
                self.cells[row][col].set_under_attack(False)
            else:
                solved = False
                self.cells[row][col].set_under_attack(True)

        return solved


def run(n=5):
    # Run the game logic
    nqueens_game = NQueensGame(n)

    # Set the display
    surface = pygame.display.set_mode((nqueens_game.width, nqueens_game.height))
    pygame.display.set_caption('N-Queens')

    # Clock to control fps
    clock = pygame.time.Clock()

    # Create the "Solved" text to display when the goal is reached
    solve_font = pygame.font.SysFont("sanscomic", int(nqueens_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (nqueens_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (nqueens_game.height / 2) - (solve_text.get_height() / 2)

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            # Get the click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                nqueens_game.click_cell_at_pos(event.pos)

        # Draw the board
        nqueens_game.draw(surface)

        # Display the "Solved" text if the goal is reached
        if nqueens_game.solved():
            surface.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()
        clock.tick(60)


# Entry point of the script
if __name__ == '__main__':
    if len(sys.argv) == 1:
        run()
    else:
        try:
            n = int(sys.argv[1])
            run(n)
        except ValueError:
            print('Usage: python nqueens_pygame.py [n]')
