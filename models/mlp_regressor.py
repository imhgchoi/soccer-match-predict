import numpy as np
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler

class MLPRegressor(BaseModel):
    def __init__(self, config, dataset):
        super(MLPRegressor, self).__init__(config, dataset)