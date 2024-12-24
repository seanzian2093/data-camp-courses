from node import Node


class LinkedList:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def remove_at_beginning(self):
        # Update the head to next of orignal head
        self.head = self.head.next

    def insert_at_beginning(self, data):
        # create a new node from data
        new_node = Node(data)
        # if head is not None, set the next of new node to head
        # and update the head with new node
        if self.head:
            new_node.next = self.head
            self.head = new_node
        # if head is None, set the head and tail to new node, i.e., the LinkedList is empty
        else:
            self.tail = new_node
            self.head = new_node
