def insertion_sort(my_list):
    for i in range(1, len(my_list)):
        number_to_order = my_list[i]
        j = i - 1
        while j >= 0 and number_to_order < my_list[j]:
            my_list[j + 1] = my_list[j]
            j -= 1
        my_list[j + 1] = number_to_order
    return my_list


if __name__ == "__main__":
    print(insertion_sort([4, 3, 7, 1, 5]))
