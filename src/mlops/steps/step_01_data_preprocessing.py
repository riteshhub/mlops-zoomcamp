from mlops.config.config_manager import ConfigurationManager
from mlops.components.data_preprocessing import DataPreProcessing
from mlops.utils.mlflow_utils import mlflow_run_manager, log_to_mlflow
from mlops import logger
from pathlib import Path
import pandas as pd
import numpy as np

STAGE_NAME = "Data PreProcessing stage"

class DataPreProcessingStep:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            mlflow_run_manager(
                mlflow_experiment_name=config_manager.config.mlflow_experiment_name,
                mlflow_experiment_run_name_mode=config_manager.config.mlflow_experiment_run_name_mode,
                child_run_name="preprocess",
                experiment_tracker_url=config_manager.config.mlflow_uri,
                start=True,
            )
            data_preprocessing_config = config_manager.get_data_preprocessing_config()
            data_preprocessing = DataPreProcessing(config=data_preprocessing_config)
            df = pd.read_csv(data_preprocessing_config.input_data_path)
            logger.info('input data successfully loaded')

            y = df.pop(data_preprocessing_config.all_schema.TARGET_COLUMN.name)
            y_pre = y.to_numpy().reshape(len(y), 1)
            transformer = data_preprocessing.perform_preprocessing(df)
            X_pre = transformer.transform(df)
            X = np.concatenate((y_pre, X_pre), axis=1)
            
            data_preprocessing.write_processed_data(X, data_preprocessing_config.train_validation_test_split)
            log_to_mlflow(step=data_preprocessing_config.step_name, configuration=config_manager)

        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataPreProcessingStep()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

