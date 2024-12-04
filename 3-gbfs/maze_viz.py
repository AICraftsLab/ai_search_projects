import sys
import pygame
import random
from frontier import PriorityQueueFrontier
from node import Node

pygame.init()


class Maze:
    def __init__(self, filepath):
        """
        Initializes a maze object from a file.

        Args:
            filepath: The path of the file containing the maze.
        """

        # Read the file
        with open(filepath) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Splitting the maze file. splitlines() will return
        # a list of the lines in the maze file.
        contents = contents.splitlines()

        # Determine height and width of the maze
        self.height = len(contents)  # Number of lines in the file
        self.width = max(len(line) for line in contents)  # Number of chars in the longest line

        # Keep track of walls. Create a 2D list of bools to
        # represent the places where there is a wall in the maze
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)  # Saving the start position
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)  # Saving the goal position
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

    def actions(self, state):
        """
        Returns the valid actions from a given state.

        Args:
            state: The current state in the maze.

        Returns:
            A list of valid actions (up, down, left, right).
        """
        actions = []
        row, col = state  # Current state row and col

        # Action is available if it will not lead to a wall, or outside the maze.
        if row - 1 >= 0 and not self.walls[row - 1][col]:  # Checking availability of moving up
            actions.append('up')
        if row + 1 < self.height and not self.walls[row + 1][col]:  # Checking availability of moving down
            actions.append('down')
        if col - 1 >= 0 and not self.walls[row][col - 1]:  # Checking availability of moving left
            actions.append('left')
        if col + 1 < self.width and not self.walls[row][col + 1]:  # Checking availability of moving right
            actions.append('right')

        random.shuffle(actions)
        return actions

    def result(self, state, action):
        """
        Returns the resulting state after taking an action in a state.

        Args:
            state: The current state in the maze.
            action: The action to take from the current state.

        Returns:
            The resulting state after taking the action.
        """
        row, col = state  # Current state row and col
        new_state = None

        if action == 'up':
            new_state = (row - 1, col)
        elif action == 'down':
            new_state = (row + 1, col)
        elif action == 'right':
            new_state = (row, col + 1)
        elif action == 'left':
            new_state = (row, col - 1)

        return new_state

    def get_solution(self, node):
        """
       Returns the solution path from the root node to the given node (goal node).

       Args:
           node: The node representing the goal state.

       Returns:
           The sequence of actions representing the solution path.
       """
        solution = []  # List for the sequence of the actions.

        # Loop will stop at the root node which has 'None' as parent
        while node.parent is not None:
            solution.append(node.action)
            node = node.parent

        # The sequence of actions is from goal node to root node.
        # Reverse it to be from root node to goal node
        solution.reverse()

        return solution

    def print_solved(self, solution):
        """
        Prints the maze with the solution path marked.

        Args:
            solution: The sequence of actions representing the solution path.
        """
        row, col = self.start  # Start row and col

        solution_coordinates = []

        # Getting coordinates of the solution path starting from the starting position
        for action in solution:
            if action == 'up':
                row -= 1
                solution_coordinates.append((row, col))
            elif action == 'down':
                row += 1
                solution_coordinates.append((row, col))
            elif action == 'right':
                col += 1
                solution_coordinates.append((row, col))
            elif action == 'left':
                col -= 1
                solution_coordinates.append((row, col))

        # Print the maze with solution path marked by '*'
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:  # If is a wall
                    print("#", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif (i, j) in solution_coordinates:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()


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

    # Mark this cell as part of the path to the goal
    def set_as_path(self):
        if not self.is_wall and not self.is_goal and not self.is_start:
            self.color = 'yellow'

    # Mark this cell as explored
    def set_as_explored(self):
        if not self.is_wall and not self.is_goal and not self.is_start:
            self.color = 'darkkhaki'


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
def run(maze_filepath):
    # Initialize the maze game and maze solver
    maze_game = MazeGame(maze_filepath)
    maze = Maze(maze_filepath)

    # Set Node class-level variables
    Node.goal_state = maze.goal

    # Create the initial node with the start position
    start = Node(maze.start, None, None, 0)
    frontier = PriorityQueueFrontier()
    frontier.add(start)

    explored = []
    solution = None
    solution_index = 0

    # Setup Pygame display
    pygame.display.set_caption('Maze Solver')
    surface = pygame.display.set_mode((maze_game.width, maze_game.height))
    clock = pygame.time.Clock()
    fps = 5

    # Create font for "Solved" text
    solve_font = pygame.font.SysFont("comicsans", int(maze_game.width / 5))
    solve_text = solve_font.render('Solved', 1, 'blue')
    solve_text_x = (maze_game.width / 2) - (solve_text.get_width() / 2)
    solve_text_y = (maze_game.height / 2) - (solve_text.get_height() / 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        # If solution exists, follow it step by step
        # else, search for the solution
        if solution is not None:
            # If it has finished following the solution steps
            if solution_index == len(solution):
                continue

            # Move in the direction specified by the solution
            maze_game.move(solution[solution_index])

            # Mark the cell as part of the path to the goal
            maze_game.current_cell.set_as_path()

            solution_index += 1

            # Draw the current state of the maze
            maze_game.draw(surface)

            # Display "Solved" text if the goal is reached
            if maze_game.solved():
                surface.blit(solve_text, (solve_text_x, solve_text_y))

        else:
            # If nothing left in frontier, no solution exists
            if frontier.is_empty():
                raise Exception("no solution")
            # else:
            #     print(f'{len(frontier.nodes)} nodes in frontier')

            # Get a node from the frontier
            node = frontier.pop()

            # If the node is the goal, print the solution
            if node.state == maze.goal:
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                solution = maze.get_solution(node)
                maze.print_solved(solution)
                print(solution)
                continue

            # Add node to list of explored nodes
            explored.append(node.state)

            # Mark the node's cell as explored
            row = node.state[0]
            col = node.state[1]
            maze_game.cells[row][col].set_as_explored()

            # Expand the node
            for action in maze.actions(node.state):
                new_state = maze.result(node.state, action)
                new_node = Node(new_state, node, action, node.cost + 1)

                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)

            maze_game.draw(surface)

        # Update the display
        pygame.display.flip()
        clock.tick(fps)


# Entry point of the script
if __name__ == '__main__':
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        run('../mazes/maze0.txt')
