def binary_search(ordere_list, search_value):
    first = 0
    last = len(ordere_list) - 1

    while first <= last:
        middle = (first + last) // 2

        if ordere_list[middle] == search_value:
            return True
        elif search_value < ordere_list[middle]:
            last = middle - 1
        else:
            first = middle + 1
    return False


def binary_search_recursive(ordere_list, search_value):
    # Define the base case
    if len(ordere_list) == 0:
        return False
    else:
        middle = len(ordere_list) // 2
        # Check if the middle value is the search value
        if ordere_list[middle] == search_value:
            return True
        # If not, call recursively the function with the right half of the list
        elif search_value < ordere_list[middle]:
            return binary_search_recursive(ordere_list[:middle], search_value)
        # If not, call recursively the function with the left half of the list
        else:
            return binary_search_recursive(ordere_list[middle + 1 :], search_value)


if __name__ == "__main__":
    print(binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  # True
    print(binary_search_recursive([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  # True
