from mlops.config.config_manager import ConfigurationManager
from mlops.components.model_registry import ModelRegistry
from mlops.utils.mlflow_utils import (mlflow_run_manager, 
                                             log_to_mlflow, 
                                             mlflow_run_finder)
from mlops import logger
import mlflow

STAGE_NAME = "Model Registry stage"

class ModelRegistryStep:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()

            run = mlflow_run_manager(
                mlflow_experiment_name=config_manager.config.mlflow_experiment_name,
                mlflow_experiment_run_name_mode=config_manager.config.mlflow_experiment_run_name_mode,
                child_run_name="register",
                experiment_tracker_url=config_manager.config.mlflow_uri,
                start=False,
            )

            experiment = mlflow.get_experiment_by_name(config_manager.config.mlflow_experiment_name)
            experiment_id = experiment.experiment_id
            parent_run_name = run.data.params["its_parent_run_name"]

            run_df = mlflow_run_finder(
                experiment_id,
                "train",
                run_type="child",
                parent_run_name=parent_run_name,
            )

            model_registry_config = config_manager.get_model_registry_config()
            model_registry = ModelRegistry(config=model_registry_config)
            model_registry.register(model_registry_config.register_model_name, run_df.run_id, config_manager.config.mlflow_uri)
            log_to_mlflow(step=model_registry_config.step_name, configuration=config_manager)
            mlflow.end_run()

        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelRegistryStep()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

