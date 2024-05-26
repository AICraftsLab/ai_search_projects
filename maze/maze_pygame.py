import sys
import pygame

pygame.init()


# Define the Cell class to represent each cell in the maze
class Cell:
    size = 30  # Cells size
    border_width = 1  # Cells border width

    def __init__(self, row, col, is_wall=False, is_start=False, is_goal=False):
        self.row = row  # Row number
        self.col = col  # Column number
        self.x = col * Cell.size  # X pos
        self.y = row * Cell.size  # Y pos
        self.is_wall = is_wall
        self.is_goal = is_goal
        self.is_start = is_start

        # Set the cell color based on its type
        self.color = 'white'  # empty space color
        if is_wall:
            self.color = 'dimgray'
        elif is_start:
            self.color = 'red'
        elif is_goal:
            self.color = 'blue'

    def draw(self, surface, is_current=False):
        # Draw the outer rectangle (cell border)
        pygame.draw.rect(surface, 'black', (self.x, self.y, Cell.size, Cell.size))  # Border color: black

        # Draw the inner rectangle (cell itself) with the appropriate color
        inner_size = Cell.size - (Cell.border_width * 2)
        color = self.color if not is_current else 'green'  # Current cell color: green
        pygame.draw.rect(surface, color,
                         (self.x + Cell.border_width, self.y + Cell.border_width, inner_size, inner_size))


# Define the MazeGame class to manage the maze and game logic
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

        # Determine number of rows and cols in the maze
        self.rows = len(contents)
        self.cols = max(len(line) for line in contents)

        # Determine size of the game window
        self.width = self.cols * Cell.size
        self.height = self.rows * Cell.size

        # 2D array of all the cells in the maze
        self.cells = []

        # Keep track of the current cell
        self.current_cell = None

        # Initialize the maze cells
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
        # Get current cell row and column
        row, col = self.current_cell.row, self.current_cell.col

        neighbor = None  # Destination cell

        # Determine the neighboring cell based on the direction
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

        # Move to the neighbor cell if it's not a wall or outside the maze
        if neighbor is not None and not neighbor.is_wall:
            self.current_cell = neighbor  # Updating current cell

    def solved(self):
        # Check if the current cell is the goal
        return self.current_cell.is_goal

    def draw(self, surface):
        # Draw all cells in the maze
        for row in self.cells:
            for cell in row:
                if cell is self.current_cell:
                    cell.draw(surface, True)
                else:
                    cell.draw(surface)


# Function to run the maze game
def run(filepath):
    maze_game = MazeGame(filepath)

    pygame.display.set_caption('Maze')  # Game window caption
    surface = pygame.display.set_mode((maze_game.width, maze_game.height))  # Window width and height

    # Create the "Solved" text to display when the goal is reached
    solve_font = pygame.font.SysFont("sanscomic", int(maze_game.width / 4))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (maze_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (maze_game.height / 2) - (solve_text.get_height() / 2)

    # Main game loop
    while True:
        for event in pygame.event.get():
            # Quit game on closing game window
            if event.type == pygame.QUIT:
                quit()

            # Get the pressed key and move accordingly
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                maze_game.move('up')
            elif keys[pygame.K_RIGHT]:
                maze_game.move('right')
            elif keys[pygame.K_DOWN]:
                maze_game.move('down')
            elif keys[pygame.K_LEFT]:
                maze_game.move('left')

        # Draw the maze and the current state
        maze_game.draw(surface)

        # Display the "Solved" text if the goal is reached
        if maze_game.solved():
            surface.blit(solve_text, (solve_text_x, solve_text_y))

        # Update the display
        pygame.display.flip()


# Entry point of the script
if __name__ == '__main__':
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        run('maze0.txt')
