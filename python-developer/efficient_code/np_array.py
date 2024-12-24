import numpy as np

# Create a list by unpacking a range
arrival_times = [*range(10, 60, 10)]

# Convert the arrival_times list to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3
print(new_times)
