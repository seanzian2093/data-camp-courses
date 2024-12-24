from abc import ABC, abstractmethod


# An abstract class
class Company(ABC):
    # an abstrct method that must be implemented by subclasses
    @abstractmethod
    def create_budget(self):
        pass

    # a concrete method that can be inherited by subclasses
    def hire_employee(self, name):
        print(f"Employee {name} hired")


# A subclass of Company
class Technology(Company):
    def __init__(self, name) -> None:
        self.name = name

    def create_budget(self, year, expenses):
        for expense, amount in expenses.items():
            print(f"{year} budget for {expense} is {amount}")


if __name__ == "__main__":
    t = Technology("Tina's Tech Advisors")
    t.create_budget(2021, {"salaries": 100000, "rent": 20000})
    t.hire_employee("John Doe")
