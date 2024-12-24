# Absolute import
# from mysklearn.proprocessing import normalize

# Relative import
from . import normalize

# Import function to subpackage - so that we can use it as mysklearn.proprocessing.normalize_data
from .normalize import normalize_data

# from mysklearn.proprocessing.normalize import normalize_data
