# FIFO Queue - first in first out queue
# Python's Queue module provides a FIFO queue implementation. - but we are going to implement our own FIFO queue
# from queue import Queue
from node import Node


class Queue:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def enqueue(self, data):
        new_node = Node(data)
        # Check if the queue is empty by checking if the head is None
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        # If the queue is not empty, add the new node by
        # setting the next of the old tail to the new node, i.e. each non-tail node has a non-None next node
        # and updating the tail with new node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        # Check if the queue's head is not None
        if self.head:
            # Assign the current head to a variable
            current_node = self.head
            # Update the head with the next of the current head
            self.head = current_node.next
            # Set the next of the variable to None
            current_node.next = None
        # After update the head, check if the queue is empty, i.e. head is None
        if self.head is None:
            # Update the tail to None
            self.tail = None
        return current_node.data

    def has_elements(self):
        return self.head is not None


class PrinterTasks:
    def __init__(self) -> None:
        self.queue = Queue()

    def add_document(self, document):
        self.queue.enqueue(document)

    def print_documents(self):
        while self.queue.has_elements():
            print("Printing", self.queue.dequeue())


if __name__ == "__main__":
    printer = PrinterTasks()
    printer.add_document("Python Basics")
    printer.add_document("Python Advanced")
    printer.add_document("Python Data Structures")
    printer.print_documents()
