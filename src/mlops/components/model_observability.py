from mlops import logger
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from mlops.config.config_steps import ModelObservabilityConfig
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.metrics import *

import pandas as pd


class ModelObservability:
    
    def __init__(self, config: ModelObservabilityConfig):
        """initilize class

        Args:
            config (ModelObservabilityConfig): object
        """
        self.config = config

    def prepare_monitoring_report(self, transformer:ColumnTransformer, model:XGBClassifier, production_data:pd.DataFrame):
        try:
            target = self.config.all_schema.TARGET_COLUMN.name
            numerical_features = list(self.config.all_schema.NUMERICAL_FEATURES)
            categorical_features = list(self.config.all_schema.CATEGORICAL_FEATURES)
            column_mapping = ColumnMapping()
            column_mapping.target = target
            column_mapping.prediction = 'prediction'
            column_mapping.numerical_features = numerical_features
            column_mapping.categorical_features = categorical_features
            
            reference_data = pd.read_csv(self.config.input_data_path)
            y = reference_data.pop(target)
            logger.info("preparing reference data")
            transformed_data = transformer.transform(reference_data)
            reference_data['prediction'] = model.predict(transformed_data)

            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                TargetDriftPreset()
            ])
            report.run(reference_data=reference_data, current_data=production_data)
            logger.info(f"saving report at path {self.config.report_path}")
            ts = (report.timestamp).strftime('%Y-%m-%d_%H-%M-%S')
            report.save_json(f"{self.config.report_path}/report_{ts}.json")
            report.save_html(f"{self.config.report_path}/report_{ts}.html")
        except Exception as e:
            raise e