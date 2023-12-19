import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class DigitRecogniser:

    def __init__(self, filePath: str) -> None:

        self.data = np.array(pd.read_csv(filePath))
        self.m, self.n = self.data.shape # m is the number of observations, n is the number of features + 1 (label)
        np.random.shuffle(self.data)

        vldData = self.data[0:1000].T
        self.vldLabels = vldData[0]
        self.vldInputs = vldData[1:self.n]/255.

        trnData = self.data[1000:self.m].T
        self.trnLabels = trnData[0]
        self.trnInputs = trnData[1:self.n]/255.