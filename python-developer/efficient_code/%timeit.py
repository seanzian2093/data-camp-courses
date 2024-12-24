# In IPython shell, use `%timeit -r5 -n25` to run the code 5 times with 25 loops each time

# Create a list - list comprehension vs. unpacking a range - unpacking a range is faster
# `%timeit [i for i in range(1000)]`
# `%timeit [*range(1000)]`

# Create a tuple/list/dict - formal name vs. literal syntax - literal syntax is faster
# `%timeit lst = list()`
# `%timeit lst = []`

# `%timeit lst = tuple()`
# `%timeit lst = ()`

# `%timeit lst = dict()`
# `%timeit lst = {}`
