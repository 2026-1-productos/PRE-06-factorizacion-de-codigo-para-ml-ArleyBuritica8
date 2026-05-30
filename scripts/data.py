import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 123456
URL= "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FOLDER = "data"

def prepare_data(url):
    df = pd.read_csv(url, sep=";")
    y = df["quality"]
    x = df.copy()
    x.pop("quality")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return x_train, x_test, y_train, y_test

def main():
    x_train, x_test, y_train, y_test=prepare_data(URL)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    x_train.to_csv(os.path.join(DATA_FOLDER,"x_train.csv"), index=False)
    x_test.to_csv(os.path.join(DATA_FOLDER, "x_test.csv"), index=False) 
    y_train.to_csv(os.path.join(DATA_FOLDER, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_FOLDER, "y_test.csv"), index=False)
if __name__ == "__main__":
    main()