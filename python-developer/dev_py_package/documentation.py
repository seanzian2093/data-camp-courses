"""
A module for generating documentation for the dev_py_package package.
"""

from typing import List


def count_words(file_path: str, words_list: List[str]) -> int:
    """Count the number of words in a text file.
    
    This function reads the text from a file and counts the number of words.

    Parameters
    ----------
    file_path : str : The path to the text file.
        
    words_list : List[str] A list of words to search for in the text.
        
    file_path : str :
        
    words_list : List[str] :
        
    file_path : str :
        
    words_list : List[str] :
        
    file_path: str :
        
    words_list: List[str] :
        

    Returns
    -------

    
    """
    with open(file_path) as file:
        text = file.read()
    return len(text.split())


# Use `pyment` to generate docstrings.
# run `pip install pyment`
# run `pyment -w -o numpy dev_py_package/documentation.py`
# `-w` flag writes the changes to the file
# `-o numpy` flag specifies the format of the docstrings

# run `pyment -w -o numpy .` to generate docstrings for all files in the current directory
