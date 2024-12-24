import queue
from node import TreeNode


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        new_node = TreeNode(data)
        # If the tree is empty, set the new node as the root
        if self.root is None:
            self.root = new_node
            return
        else:
            # if not empty, start from the root
            current_node = self.root
            while True:
                # If the data is less than the current node data, insert to the left
                if data < current_node.data:
                    # if left_child is None, update it with new_node
                    if current_node.left_child is None:
                        current_node.left_child = new_node
                        return
                    # otherwise, update the current_node to the left_child for next iteration
                    else:
                        current_node = current_node.left_child
                # Similar case when data is greater than the current node data
                elif data > current_node.data:
                    if current_node.right_child is None:
                        current_node.right_child = new_node
                        return
                    else:
                        current_node = current_node.right_child

    def search(self, search_value):
        # Start from top of the tree
        current_node = self.root
        # While there is a node
        while current_node:
            # If the search value is equal to the current node value
            if search_value == current_node.data:
                return True
            # If the search value is less than the current node value
            elif search_value < current_node.data:
                current_node = current_node.left_child
            # If the search value is greater than the current node value
            else:
                current_node = current_node.right_child
        return False

    def delete(self, delete_value):
        # no child - delete
        # one child - delete the noe and connect the parent to the child
        # two children - replace the node with its successor - the smallest value in its right child
        # find successor - go through right child and find the last, i.e. smallest value
        pass

    def find_min(self):
        current_node = self.root
        # left child is always smaller than the parent so search to last
        while current_node.left_child:
            current_node = current_node.left_child
        return current_node.data

    def in_order_traversal(self, current_node):
        """Retrieve the data in the tree in ascending order"""
        if current_node is not None:
            self.in_order_traversal(current_node.left_child)
            print(current_node.data)
            self.in_order_traversal(current_node.right_child)

    def pre_order_traversal(self, current_node):
        """Retrieve the data in the tree in pre-order: up to down, left to right"""
        if current_node is not None:
            print(current_node.data)
            self.pre_order_traversal(current_node.left_child)
            self.pre_order_traversal(current_node.right_child)

    def post_order_traversal(self, current_node):
        """Retrieve the data in the tree in post-order: left to right, up to down"""
        if current_node is not None:
            self.post_order_traversal(current_node.left_child)
            self.post_order_traversal(current_node.right_child)
            print(current_node.data)

    def breadth_first_traversal(self):
        # Check if the tree is not empty - if not empty, start from the root
        if self.root is not None:
            visited_nodes = []
            bfs_queue = queue.SimpleQueue()
            bfs_queue.put(self.root)
        while not bfs_queue.empty():
            current_node = bfs_queue.get()
            visited_nodes.append(current_node.data)
            if current_node.left_child is not None:
                bfs_queue.put(current_node.left_child)
            if current_node.right_child is not None:
                bfs_queue.put(current_node.right_child)

        return visited_nodes


if __name__ == "__main__":
    # bst = BinarySearchTree()
    # bst.insert("Pride and Prejudice")
    # print(bst.search("Pride and Prejudice"))  # True

    bst = BinarySearchTree()
    bst.insert("Little Women")
    bst.insert("Heidi")
    bst.insert("Oliver Twist")
    bst.insert("Dracula")
    bst.insert("Jane Eyre")
    bst.insert("Moby Dick")
    bst.insert("Vanity Fair")
    # print(bst.find_min())

    # print(bst.in_order_traversal(bst.root))
    # print(bst.pre_order_traversal(bst.root))
    # print(bst.post_order_traversal(bst.root))

    print(bst.breadth_first_traversal())
