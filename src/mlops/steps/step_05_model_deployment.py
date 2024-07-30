from mlops.config.config_manager import ConfigurationManager
from mlops.components.data_preprocessing import DataPreProcessing
from mlops.components.model_deployment import ModelDeployment
from mlops.components.model_observability import ModelObservability
from mlops.utils.mlflow_utils import mlflow_run_finder, mlflow_run_manager
from mlops import logger
from fastapi import FastAPI
import uvicorn
import mlflow

STAGE_NAME = "Model Deployment stage"
HOST = 'localhost'
PORT=8001

class ModelDeploymentStep:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            api_client = FastAPI()

            run = mlflow_run_manager(
                mlflow_experiment_name=config_manager.config.mlflow_experiment_name,
                mlflow_experiment_run_name_mode=config_manager.config.mlflow_experiment_run_name_mode,
                child_run_name="deploy",
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

            model_deployment_config = config_manager.get_model_deployment_config()
            model_deployment = ModelDeployment(config=model_deployment_config)
            model_observability_config = config_manager.get_model_observability_config()
            model_observability = ModelObservability(config=model_observability_config)

            transformer = model_deployment.load_transformed_model(experiment_tracker_url=config_manager.config.mlflow_uri, run_id=run_df.run_id)
            model = model_deployment.load_trained_model(model_name=model_deployment_config.register_model_name, experiment_tracker_url=config_manager.config.mlflow_uri)
            model_deployment.create_endpoint_route(model_observability=model_observability ,client=api_client, model=model, transformer=transformer)
            mlflow.end_run()
            uvicorn.run(api_client, host=HOST, port=PORT)

        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelDeploymentStep()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e