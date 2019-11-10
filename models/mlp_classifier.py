import numpy as np
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler

class MLPClassifier(BaseModel):
    def __init__(self, config, dataset):
        super(MLPClassifier, self).__init__(config, dataset)