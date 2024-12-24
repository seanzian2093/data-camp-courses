def partition(my_list, first_index, last_index):
    pivot = my_list[first_index]
    left_pointer = first_index + 1
    right_pointer = last_index

    while True:
        while my_list[left_pointer] < pivot and left_pointer < last_index:
            left_pointer += 1
        while my_list[right_pointer] > pivot and right_pointer >= first_index:
            right_pointer -= 1
        if left_pointer >= right_pointer:
            break

        my_list[left_pointer], my_list[right_pointer] = (
            my_list[right_pointer],
            my_list[left_pointer],
        )

    my_list[first_index], my_list[right_pointer] = (
        my_list[right_pointer],
        my_list[first_index],
    )
    return right_pointer


def quick_sort(my_list, first_index, last_index):
    if first_index < last_index:
        partition_index = partition(my_list, first_index, last_index)
        quick_sort(my_list, first_index, partition_index)
        quick_sort(my_list, partition_index + 1, last_index)


if __name__ == "__main__":
    my_list = [6, 2, 9, 7]
    quick_sort(my_list, 0, len(my_list) - 1)
    print(my_list)
