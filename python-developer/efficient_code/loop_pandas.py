# Use `df.iterrows()` to iterate over rows in a DataFrame, yielding index and Series of row pairs

# Use `df.itertuples()` to iterate over rows in a DataFrame, yielding a named tuple of index and Series of row pairs
## with name attribute for each column
## this is much faster than `df.iterrows()`
## use dot notation to access the column values


# Use `df.apply()` to apply a function along the axis of a DataFrame, to avoid using a loop
## axis=0 for column-wise operation
## axis=1 for row-wise operation

# Use `Series.values` to perform element-wise operation on a Series, much faster than using a loop
# or using `Series` directly or using `.loc` or `.iloc` to access the elements in loop
