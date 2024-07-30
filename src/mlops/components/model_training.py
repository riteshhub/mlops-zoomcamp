import pandas as pd
import os
from mlops import logger
from xgboost import XGBClassifier
from mlops.config.config_steps import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        """initilize class

        Args:
            config (ModelTrainingConfig): object
        """
        self.config = config

    
    def train(self, datasets: list):
        """train model with hyperparameters and save model file

        Returns:
            None
        """
        xgb = XGBClassifier(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, learning_rate=self.config.learning_rate,objective=self.config.objective)
        
        for data in datasets:
            x = data.select(data.columns[1:len(data.columns)]).toPandas().values
            y = data.select(data.columns[0]).toPandas().values
            xgb.fit(x, y)

        return xgb

