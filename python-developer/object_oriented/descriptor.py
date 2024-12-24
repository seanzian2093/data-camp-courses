from typing import Any


class BankAccount:
    def __init__(self, email):
        self.email = email

    @property
    def email(self):
        # how `self._email` is used here is a convention to avoid name clashes
        return f"Email for this account is: {self._email}"

    @email.setter
    def email(self, new_email_address):
        if not "@" in new_email_address:
            print(f"Invalid email address: {new_email_address}")
        self._email = new_email_address

    @email.deleter
    def email(self):
        del self._email
        print("Email deleted, make sure to add a new email!")


if __name__ == "__main__":
    account = BankAccount("example@abc.com")
    print(account.email, "\n")
    account.email = "example@bcd.com"
    print(account.email, "\n")
    del account.email

    account.balance = 100
    account.beneficiary = "John Doe"
