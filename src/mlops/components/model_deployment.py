from mlops.config.config_steps import ModelDeploymentConfig, LoanDetails
from mlops.components.model_observability import ModelObservability 
from mlops.utils.mlflow_utils import MlflowClientManager
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from mlops import logger
import pandas as pd
import numpy as np
from fastapi import FastAPI
import mlflow


class ModelDeployment:
    def __init__(self, config: ModelDeploymentConfig):
        """initilize class

        Args:
            config (ModelDeploymentConfig): object
        """
        self.config = config
    
    def load_trained_model(self, model_name:str, experiment_tracker_url:str, stage:str="production") -> XGBClassifier:
        """load latest production staged model from model registry

        Returns:
            None
        """
        mlflow.set_tracking_uri(experiment_tracker_url) 
        client = MlflowClientManager()
        model_metadata = dict(client.load_model_details(model_name, stages=[stage]))
        logger.info(f"model info : {model_metadata}")
        latest_model_run_id = model_metadata['run_id']

        logger.info(f"model loaded successfully from run id {latest_model_run_id}")
        model = mlflow.xgboost.load_model(f"runs:/{latest_model_run_id}/model")

        return model
    
    def load_transformed_model(self, experiment_tracker_url:str, run_id:str) -> ColumnTransformer:
        """load transformer model

        Returns:
            None
        """
        mlflow.set_tracking_uri(experiment_tracker_url)
        logger.info(f"model loaded successfully from run id {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/sk_models")

        return model
    
    def create_endpoint_route(self, model_observability:ModelObservability, client:FastAPI, model:XGBClassifier, transformer: ColumnTransformer):
        @client.get("/")
        async def index():
            return {"Hello": "World"}


        @client.post("/predict")
        async def predict(data:LoanDetails):
            logger.info([data.model_dump()])
            df = pd.DataFrame([data.model_dump()])
            transformed_data = transformer.transform(df)
            prediction = model.predict(transformed_data)
            df['prediction'] = prediction
            logger.info("preparing monitoring report")
            model_observability.prepare_monitoring_report(transformer=transformer, model=model, production_data=df)
            return {"prediction": str(prediction[0])}
