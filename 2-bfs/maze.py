import sys
import random
from frontier import StackFrontier, QueueFrontier
from node import Node


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

        # Keep track of walls. Create a 2D list of bools to represent the places where
        # there is a wall in the maze. True for walls, False otherwise.
        # The maze is expected to be a rectangular maze, so IndexError will be risen when
        # the length of the lines in the maze are not equal. In that case, False will
        # be added to that row/line signifying an empty space. Generally, all short
        # lines will be filled with empty spaces to match the length of the maze
        # (i.e. the longest line) making the maze rectangular.
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
                if col:  # If is wall
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

    def search(self):
        """
        Solves the maze.
        """
        root_node = Node(self.start, None, None, 0)
        frontier = QueueFrontier()
        frontier.add(root_node)

        explored = []

        while True:
            # If nothing is left in frontier, no solution exists
            if frontier.is_empty():
                raise Exception("no solution")
            # else:
                # print(f'{len(frontier.nodes)} nodes in frontier')

            # Get a node from the frontier
            node = frontier.pop()

            # If the node is the goal, print the solution
            if node.state == self.goal:
                print("Solution Found")
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                solution = self.get_solution(node)
                self.print_solved(solution)
                print(solution)
                break

            # Add node to list of explored nodes
            explored.append(node.state)

            # Expand the node
            for action in self.actions(node.state):  # For all actions that can be taken in the state
                new_state = self.result(node.state, action)  # Get the new state after performing the action
                new_node = Node(new_state, node, action, node.cost + 1)  # Create a new node for the new state

                # If the node is not already in the frontier and not explored, add it to the frontier
                if not frontier.contains(new_node) and new_node.state not in explored:
                    frontier.add(new_node)


# Entry point of the script
if __name__ == '__main__':
    if len(sys.argv) == 2:
        maze = Maze(sys.argv[1])
        maze.search()
    else:
        maze = Maze('../mazes/maze5.txt')
        maze.search()
