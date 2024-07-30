from mlops.config.config_steps import ModelRegistryConfig
from mlops.utils.mlflow_utils import MlflowClientManager
from mlops import logger


class ModelRegistry:
    def __init__(self, config: ModelRegistryConfig):
        """initilize class

        Args:
            config (ModelRegistryConfig): object
        """
        self.config = config
    
    def register(self, model_name:str, run_id:str, uri:str) -> None:
        """register model in model registry

        Returns:
            None
        """  
        centralized_mlflow_client = MlflowClientManager()
        registered_model = centralized_mlflow_client.register_in_none_stage(
            model_name=model_name,
            run_id=run_id,
            model_registry_workspace_url=uri,
        )

        version_with_dependencies = str(
        int(registered_model.version)
        )  # a new version is registered

        # NOTE: Since we have run an acceptance criteria in the model evaluation phase,
        # it is safe to automatically promote the model to Production.
        # However, further tests can be done here if needed.
        centralized_mlflow_client.promote_to_staging(registered_model.name, version_with_dependencies)
        logger.info(f"Model {registered_model.name} promoted to Staging")

        centralized_mlflow_client.promote_to_production(registered_model.name, version_with_dependencies)
        logger.info(f"Model {registered_model.name} promoted to Production")