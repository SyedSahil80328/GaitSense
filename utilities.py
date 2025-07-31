from glob import glob
import pandas as pd

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option("display.width", 200)  # Set large enough to avoid wrapping
pd.set_option("display.max_columns", None)  # Show all columns

def return_file_paths(path):
    return sorted(glob(path))

class Formatter:
    def return_formatted_string(self, string, symbol = '-'):
        decrator = symbol * ((200 - len(string))//2 + 1) if len(string)&1 else symbol * ((200 - len(string))//2)
        return f"{decrator} {string} {decrator}"