# Best practice to name the source code folder the same as the top folder name, i.e.`mksklearn/mysklearn`
# This is the setup file for the package, should be `mysklearn/setup.py`, outside of `mysklearn/mysklearn`

# Import the setup function from the setuptools module
from setuptools import setup, find_packages

# Call the setup function with the required arguments
setup(
    author="Your Name",  # Name of the package author
    description="A small example package",  # Short description of the package
    name="mysklearn",  # Name of the package
    version="0.1.0",  # Version number
    packages=find_packages(
        # install `mysklearn` and all its sub-packages
        include=["mysklearn", "mysklearn.*"]
    ),  # List of packages to be included in the distribution
    install_requires=["numpy"],  # List of dependencies
    python_requires=">=3.6",  # Python version requirements
)

# in `mysklearn` folder, run `python setup.py sdist` to create a source distribution
# in `mysklearn` folder, run `pip install -e .` to install the package in current directory in editable mode

# use `python3 setup.py sdist bdist_wheel` to create a source distribution and a wheel distribution
# use `cookiecutter` to create a new package template
