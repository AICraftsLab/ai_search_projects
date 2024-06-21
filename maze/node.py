class Node:
    """
        Represents a node in the search tree.

        Attributes:
            goal_state: The goal state to compare against for calculating the Manhattan distance.
            as_astar: A flag to determine if the node comparison should use A* search logic.
            is_puzzle: A flag to determine if the problem is a 15-puzzle.
        """
    goal_state = None
    as_astar = False
    is_puzzle = False

    def __init__(self, state, parent, action, cost):
        """
        Initializes a node in the search tree.

        Args:
            state: The state represented by the node.
            parent: The parent node in the search tree.
            action: The action taken to reach this node from the parent node.
            cost: The cost of reaching this node from the root node.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def manhattan_distance(self):
        """
        Calculates the Manhattan distance from
        the current state to the goal state.

        Raises:
            Exception: If the goal state is not assigned.

        Returns:
            The Manhattan distance from the current state to the goal state.
        """
        if Node.goal_state is None:
            raise Exception('Goal state is not assigned')

        if Node.is_puzzle:
            if len(Node.goal_state) != 4:
                raise Exception('Goal state does not match that of puzzle')
            else:
                distance = self.puzzle_manhattan_distance()
        else:
            if len(Node.goal_state) != 2:
                raise Exception('Goal state does not match that of maze')
            else:
                distance = self.maze_manhattan_distance()

        return distance

    def maze_manhattan_distance(self):
        """
        Calculates the Manhattan distance of a Maze's
        current state to the goal state.

        Returns:
            The Manhattan distance from the current state to the goal state.
        """

        row_diff = abs(Node.goal_state[0] - self.state[0])
        col_diff = abs(Node.goal_state[1] - self.state[1])
        distance = row_diff + col_diff

        return distance

    def puzzle_manhattan_distance(self):
        """
        Calculates the Manhattan distance of a 15-Puzzle's
        current state to the goal state.

        Returns:
            The Manhattan distance from the current state to the goal state.
       """

        distance = 0

        # Iterate through each tile in the current state
        for i, row in enumerate(self.state):
            for j, num in enumerate(row):
                # Iterate through each tile in the goal state
                # to find the goal position of the current tile
                for k, goal_state_row in enumerate(Node.goal_state):
                    if num in goal_state_row:
                        # Find the column (l) in the goal state
                        # where the current number is located
                        l = goal_state_row.index(num)
                        row_diff = abs(i - k)
                        col_diff = abs(j - l)
                        distance += row_diff + col_diff
                        break

        return distance

    def __lt__(self, other):
        """
        Compares two nodes for priority queue operations. If A* search logic is enabled,
        the comparison is based on the sum of the Manhattan distance and the cost. Otherwise,
        it is based only on the Manhattan distance.

        Args:
            other (Node): Another node to compare with.

        Returns:
            bool: True if this node has a lower priority than the other node, False otherwise.
                """
        if Node.as_astar:
            return self.manhattan_distance() + self.cost < other.manhattan_distance() + other.cost
        else:
            return self.manhattan_distance() < other.manhattan_distance()