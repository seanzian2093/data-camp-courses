"""Must first install pytest-benchmark using pip install pytest-benchmark"""

import pytest


def create_list():
    return [i for i in range(1000)]


def create_set():
    return set([i for i in range(1000)])


def find(it, el=50):
    return el in it


# Performance test function - for a list
def test_list(benchmark):
    benchmark(find, it=create_list())


# Performance test function - for a set
def test_set(benchmark):
    benchmark(find, it=create_set())


# Performance test decorator - for a list
def test_list2(benchmark):
    @benchmark
    def iterate_a_list():
        for i in [i for i in range(1000)]:
            pass


# Performance test decorator - for a set
def test_set2(benchmark):
    @benchmark
    def iterate_a_set():
        for i in {i for i in range(1000)}:
            pass
