# we can import package `mysklearn` in this file without installing it because we are in the same directory
import mysklearn

# without package import, we can not access sub-packages of `mysklearn`
# import mysklearn.proprocessing in top level `__init__.py` file of `mysklearn`
help(mysklearn.proprocessing)

# without package import, we can not access module `normalize` of sub-packages `proprocessing` of `mysklearn`
# import mysklearn.proprocessing in top level `__init__.py` file of `proprocessing`
help(mysklearn.proprocessing.normalize)

# with importing function `normalize_data` in `__init__.py` file of `proprocessing`
help(mysklearn.proprocessing.normalize_data)
