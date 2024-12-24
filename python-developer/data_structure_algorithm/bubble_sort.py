def bubble_sort(my_list):
    list_length = len(my_list)
    for i in range(list_length - 1):
        for j in range(list_length - i - 1):
            if my_list[j] > my_list[j + 1]:
                my_list[j], my_list[j + 1] = my_list[j + 1], my_list[j]

    return my_list


def bubble_sort_perf(my_list):
    list_length = len(my_list)
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for j in range(list_length - 1):
            if my_list[j] > my_list[j + 1]:
                my_list[j], my_list[j + 1] = my_list[j + 1], my_list[j]
                is_sorted = False

        list_length -= 1
    return my_list


if __name__ == "__main__":
    print(bubble_sort([4, 3, 7, 1, 5]))
    print(bubble_sort_perf([4, 3, 7, 1, 5]))
