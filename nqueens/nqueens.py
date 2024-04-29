import copy
import random


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

    def search(self):
        frontier = StackFrontier()

        initial_state = ([], [], [], [])
        frontier.add(Node(initial_state, None, None, 0))

        explored = []

        while True:
            if len(frontier.nodes) == 0:
                raise Exception('No solution')
            else:
                print(f'{len(frontier.nodes)} in frontier')

            node = frontier.pop()
            explored.append(node.state)

            if self.solved(node.state):
                self.print_state(node.state)
                print('Solution found')
                print(f'Cost:{node.cost}')
                print(f'Nodes explored:{len(explored)}')
                return

            for action in self.actions(node.state):
                new_state = self.result(node.state, action)
                new_node = Node(new_state, action, node, node.cost + 1)

                if not frontier.contains(node) and new_node.state not in explored:
                    frontier.add(new_node)


if __name__ == '__main__':
    nqueens = NQueens(9)
    nqueens.search()
