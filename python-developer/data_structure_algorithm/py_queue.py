import queue

# Python's LifoQueue - Last in First Out
# Create a LifoQueue with maxsize=0,i.e., infinite queue
my_book_stack = queue.LifoQueue(maxsize=0)

# Add element
my_book_stack.put("The Alchemist")
my_book_stack.put("Don Quixote")
my_book_stack.put("Journey to the West")

# Remove element
print(my_book_stack.get())
print(my_book_stack.get())
print(my_book_stack.get())

# Python's SimpleQueue - First in First Out
orders_queue = queue.SimpleQueue()
orders_queue.put("Order 1")
orders_queue.put("Order 2")
orders_queue.put("Order 3")
print("The size is: ", orders_queue.qsize())
print(orders_queue.get())
print(orders_queue.get())
print(orders_queue.get())
print("The queue is empty: ", orders_queue.empty())
