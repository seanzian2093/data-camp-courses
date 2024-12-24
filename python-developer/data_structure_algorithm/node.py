class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left_child = left
        self.right_child = right
