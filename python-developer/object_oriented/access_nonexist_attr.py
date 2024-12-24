from typing import Any


class BankAccount:
    def __init__(self, email):
        self.email = email

    # __getattr__ is called when an attribute is not found in the class namespace
    def __getattr__(self, name):
        print(f"__getattr__ called with {name}")

    # __setattr__ is called when an attribute is set on an instance
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["account_name", "balance", "email"]:
            print(f"__setattr__ called with {name} and {value}")
            self.__dict__[name] = value
        else:
            print("Invalid attribute")


if __name__ == "__main__":
    account = BankAccount("example@abc.com")
    print(account.email)

    account.email = "example@def.com"
    print(account.email)

    del account.email
    print(account.email)

    account.balance = 100
    print(account.balance)

    account.beneficiary = "John Doe"
