def selection_sort(my_list):
    list_length = len(my_list)
    for i in range(list_length - 1):
        lowest = my_list[i]
        index = i
        for j in range(i + 1, list_length):
            if my_list[j] < lowest:
                lowest = my_list[j]
                index = j
        my_list[i], my_list[index] = my_list[index], my_list[i]
    return my_list


if __name__ == "__main__":
    print(selection_sort([4, 3, 7, 1, 5]))
