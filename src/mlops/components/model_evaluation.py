import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from mlops.config.config_steps import ModelEvaluationConfig
from typing import Tuple, Dict
from xgboost import XGBClassifier


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """initilize class

        Args:
            config (ModelEvaluationConfig): object
        """
        self.config = config

    def eval_metrics(self, actual, predict)-> Tuple[float, float, float, float]:
        """contains logic to derive evaluation metrics

        Args:
            actual (np.array): actual target values
            predict (np.array): predicted target values

        Returns:
            Tuple[float, float, float, float]
        """
        f1 = f1_score(actual, predict)
        recall = recall_score(actual, predict)
        precision = precision_score(actual, predict)
        accuracy = accuracy_score(actual, predict)
        return f1, recall, precision, accuracy
    
    def evaluate(self, datasets:list, model:XGBClassifier) -> Dict:
        """evaluate model performance based on test data

        Returns:
            None
        """
        for data in datasets:
            x = data.select(data.columns[1:len(data.columns)]).toPandas().values
            y = data.select(data.columns[0]).toPandas().values

        prediction = model.predict(x)

        (f1, recall, precision, accuracy) = self.eval_metrics(y, prediction)
        
        scores = {"f1_score": f1, "recall_score": recall, "precision_score": precision, "accuracy_score": accuracy}
        # save_json(path=Path(self.config.metric_file_name), data=scores)
        return scores

    
