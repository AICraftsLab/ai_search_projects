class Node:
    """
        Represents a node in the search tree.
    """

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
