import pytest
from datetime import datetime

days_of_week = datetime.now().isoweekday()


def get_unique_values(lst):
    return list(set(lst))


def multiple_of_two(num):
    if num == 0:
        raise (ValueError)
    return num % 2 == 0


def test_numbers():
    assert multiple_of_two(4) is True, "Wrong"
    assert multiple_of_two(3) is False, "Wrong"


# Use a context manager to check if the function raises an error
def test_zero():
    with pytest.raises(ValueError):
        multiple_of_two(0)


# Use pytest.mark.xfail to the test that you expect to fail
@pytest.mark.xfail
def test_fails():
    assert multiple_of_two(3) is True, "Wrong"


# Use pytest.mark.skipif to skip a test if the condition is met
condition_string = "days_of_week == 7"


@pytest.mark.skipif(condition_string, reason="Skip the test")
def test_function():
    assert get_unique_values([1, 2, 3, 3, 4, 5]) == [1, 2, 3, 4, 5], "Wrong"
    assert get_unique_values([1, 2, 3, 3, 4, 5, 5]) == [1, 2, 3, 4, 5], "Wrong"


# run all tests in a py file - pytest intro_to_test/run_the_test.py
# run all tests in a py file in verbose mode - pytest intro_to_test/run_the_test.py -v
# run tests that match the keyword in a py file- pytest intro_to_test/run_the_test.py -k "number"
