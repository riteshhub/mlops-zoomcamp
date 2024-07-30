import mlflow
import joblib
import os
from mlops.config.config_manager import ConfigurationManager
from mlops.components.model_training import ModelTraining
from mlops.utils.mlflow_utils import mlflow_run_manager, load_dataset_from_run, mlflow_run_finder, log_to_mlflow
from mlops import logger


STAGE_NAME = "Model Training stage"

class ModelTrainingStep:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        run = mlflow_run_manager(
                    mlflow_experiment_name=config_manager.config.mlflow_experiment_name,
                    mlflow_experiment_run_name_mode=config_manager.config.mlflow_experiment_run_name_mode,
                    child_run_name=config_manager.config.model_training.step_name,
                    experiment_tracker_url=config_manager.config.mlflow_uri,
                    start=False,
        )
        experiment = mlflow.get_experiment_by_name(config_manager.config.mlflow_experiment_name)
        experiment_id = experiment.experiment_id
        parent_run_name = run.data.params["its_parent_run_name"]
        run_df = mlflow_run_finder(
            experiment_id,
            "preprocess",
            run_type="child",
            parent_run_name=parent_run_name,
        )
        datasets = []
        if 'train_data_path' in config_manager.config.data_preprocessing.keys():
            train_data = load_dataset_from_run(run_df=run_df, run_param_name='train_data_path')
            datasets.append(train_data)
        if 'validation_data_path' in config_manager.config.data_preprocessing.keys():
            validation_data = load_dataset_from_run(run_df=run_df, run_param_name='validation_data_path')
            datasets.append(validation_data)

        model_training_config = config_manager.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model = model_training.train(datasets)
        joblib.dump(model, os.path.join(model_training_config.root_dir, model_training_config.model_name))
        log_to_mlflow(step=model_training_config.step_name, configuration=config_manager, model_obj=model)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingStep()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
