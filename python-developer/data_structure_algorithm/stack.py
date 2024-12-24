from node import Node


class Stack:
    def __init__(self):
        self.top = None
        self.size = 0

    def push(self, data):
        # Create a new node with the data
        new_node = Node(data)
        # if top is not None, set the new node's next to the top
        if self.top:
            new_node.next = self.top
        # Update the top with the new node
        self.top = new_node
        # Increment the size
        self.size += 1

    def pop(self):
        # Check if the stack is empty, i.e. top is None
        if self.top is None:
            return None
        # If the stack is not empty
        else:
            # Update the return value with the top node
            popped_node = self.top
            # Decrement the size
            self.size -= 1
            # Update the top with the next of the original top
            self.top = self.top.next
            # Set the next of the popped node to None
            popped_node.next = None
            # Return the data of the popped node
            return popped_node.data
