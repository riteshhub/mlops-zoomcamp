from pyspark.sql import SparkSession
from mlops.constants import *
from mlops.utils.common import read_yaml, create_directories
from mlops.config.config_steps import (DataPreProcessingConfig, 
                                        ModelTrainingConfig, 
                                        ModelEvaluationConfig,
                                        ModelRegistryConfig,
                                        ModelDeploymentConfig,
                                        ModelObservabilityConfig)

class ConfigurationManager:
    """Manager class to intialize configs related to different pipeline steps
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_preprocessing_config(self) -> DataPreProcessingConfig:
        """Get data pre processing related configs

        Returns:
            DataPreProcessingConfig object
        """
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreProcessingConfig(
            step_name=config.step_name,
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            train_data_path=config.train_data_path,
            validation_data_path=config.validation_data_path,
            test_data_path=config.test_data_path,
            train_validation_test_split=config.train_validation_test_split,
            all_schema=self.schema
        )

        return data_preprocessing_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        """Get model training related configs

        Returns:
            ModelTrainingConfig object
        """
        config = self.config.model_training
        params = self.params.XGBClassifier
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            step_name=config.step_name,
            root_dir=config.root_dir,
            model_name = config.model_name,
            n_estimators = params.n_estimators,
            max_depth = params.max_depth,
            learning_rate = params.learning_rate,
            objective = params.objective,
            target_column = schema.name
        )

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation related configs

        Returns:
            ModelEvaluationConfig object
        """
        config = self.config.model_evaluation
        params = self.params.XGBClassifier
        acceptance_criteria = self.params.acceptance_criteria

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            step_name=config.step_name,
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            acceptance_criteria=acceptance_criteria
        )

        return model_evaluation_config
    
    def get_model_registry_config(self) -> ModelRegistryConfig:
        """Get model registry related configs

        Returns:
            ModelRegistryConfig object
        """
        config = self.config.model_registry

        model_registry_config = ModelRegistryConfig(
            step_name=config.step_name,
            register_model_name=config.register_model_name
        )

        return model_registry_config
    
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        """Get model deployment related configs

        Returns:
            ModelDeploymentConfig object
        """
        config = self.config.model_deployment

        model_deployment_config = ModelDeploymentConfig(
            step_name=config.step_name,
            register_model_name=config.register_model_name
        )

        return model_deployment_config
    
    def get_model_observability_config(self) -> ModelObservabilityConfig:
        """Get model observability related configs

        Returns:
            ModelObservabilityConfig object
        """
        config = self.config.model_observability
        schema = self.schema

        create_directories([config.root_dir])

        model_observability_config = ModelObservabilityConfig(
            project_id=config.project_id,
            dashboard_name=config.dashboard_name,
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            report_path=config.report_path,
            all_schema=schema
        )

        return model_observability_config