class StackFrontier:
    def __init__(self):
        """
        Initializes the frontier with an empty list of nodes.
        """
        self.nodes = []

    def add(self, node):
        """
        Adds a node to the frontier.

        Args:
            node: The node to be added.
        """
        self.nodes.append(node)

    def contains(self, node):
        """
        Checks if the frontier contains a specific node.

        Args:
            node: The node to check.

        Returns:
            True if the node is found in the frontier, False otherwise.
        """
        return any(n.state == node.state for n in self.nodes)

    def is_empty(self):
        """
        Checks if the frontier is empty.

        Returns:
            True if the frontier is empty, False otherwise.
        """
        return len(self.nodes) == 0

    def pop(self):
        """
        Removes and returns the last node from the frontier.

        Returns:
            The last node from the frontier.
        """
        return self.nodes.pop()


class QueueFrontier(StackFrontier):
    def pop(self):
        """
        Removes and returns the first node from the frontier.

        Returns:
            The first node from the frontier.
        """
        return self.nodes.pop(0)
