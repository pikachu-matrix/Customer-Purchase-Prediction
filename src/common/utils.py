import pandas as pd
def load_data(file_path):
    """
    the function recieves a csv file
    it retirns a dataframe.
    """
    df = pd.read_csv(file_path)
    return df