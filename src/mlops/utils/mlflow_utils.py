import json
from datetime import datetime
from typing import Optional
from typing import Tuple, Dict
import mlflow.models.utils
import pandas as pd
from pathlib import Path
from mlflow.tracking.client import MlflowClient
from mlops import logger
from mlops.utils.common import truncate_name, get_table_version, load_dataset
from mlops.config.config_manager import ConfigurationManager
from xgboost import XGBClassifier


class MlflowClientManager(MlflowClient):
    def register_in_none_stage(
        self,
        model_name: str,
        run_id: str,
        model_registry_workspace_url: str,
        path_type: str = "model",
    ):
        is_a_new_model_name = not self.is_already_registered(model_name)
        model_registered = mlflow.register_model(f"runs:/{run_id}/{path_type}", model_name)

        # if is_a_new_model_name:  # only set the permissions to new model names
        #     logger.info(f"Setting ACLs to model: {model_name} on {environment}")
        #     set_permissions_to_model(
        #         model_name=model_name,
        #         workspace_url=model_registry_workspace_url
        #     )

        return model_registered

    def is_already_registered(self, model_name: str) -> bool:
        """Checks if a model namel is already registered.

        Returns:
            bool: True if is registered, False otherwise.
        """
        all_model_versions = self.search_model_versions(f"name = '{model_name}'")
        is_registered = bool(all_model_versions)
        logger.info(f"Model {model_name} is already registered: {is_registered}")
        return is_registered

    def load_model_details(
        self, model_name: str, stages=None, version=None
    ) -> "mlflow.entities.model_registry.model_version.ModelVersion":
        """Load the model details from MLflow."""
        if stages and not isinstance(stages, list):
            stages = [stages]

        if not version:
            version = self.get_latest_versions(model_name, stages=stages)[0].version
            logger.info(f"Model version were not specified. Loading the last version: {version}")
        model_details = self.get_model_version(model_name, version)
        return model_details

    def promote_to_staging(self, model_name, version):
        if not version:
            version = self.get_latest_versions(model_name)[0].version
            logger.info(f"Model version were not specified. Loading the last version: {version}")

        logger.info(f"Promoting model {model_name} version {version} to staging")
        self.transition_model_version_stage(name=model_name, version=version, stage="staging")

    def promote_to_production(self, model_name, version):
        if not version:
            version = self.get_latest_versions(model_name, stages=["staging"])[0].version
            logger.info(f"Model version were not specified. Loading the last version: {version}")

        logger.info(f"Promoting model {model_name} version {version} to production")
        self.transition_model_version_stage(
            name=model_name, version=version, stage="production", archive_existing_versions=True
        )


def mlflow_run_manager(
    mlflow_experiment_name: str,
    mlflow_experiment_run_name_mode: str,
    child_run_name: str,
    experiment_tracker_url: str,
    start: bool = False,
):
    """
    Start an MLflow workflow, activate parent and child runs, and set tags/params as needed.
    """
    mlflow.set_tracking_uri(experiment_tracker_url)
    logger.info("############# Mlflow tracking Initiated.")

    is_new_experiment = not bool(mlflow.get_experiment_by_name(mlflow_experiment_name))

    experiment_id = mlflow.set_experiment(mlflow_experiment_name).experiment_id
    logger.info(
        f"Found MLflow experiment {mlflow_experiment_name} with experiment_id : {experiment_id}"
    )

    # if is_new_experiment:
    #     set_permissions_to_experiment(
    #         experiment_id=experiment_id,
    #         workspace_url=experiment_tracker_url,
    #         token=token,
    #         environment=environment,
    #     )

    if not start:
        parent_run_id, parent_run_name = load_last_parent_run(
            experiment_id, mlflow_experiment_run_name_mode
        )
    if start or not parent_run_id:
        parent_run_name = get_name(
            run_mode=mlflow_experiment_run_name_mode, add_date=True
        )
        mlflow.end_run()
        with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(
                f"Initiating new MLflow PARENT run '{parent_run_name}' with run_id '{parent_run_id}'"
            )
            mlflow.log_param("parent_run_name", parent_run_name)
            mlflow.autolog(log_models=False)

    mlflow.end_run()
    with mlflow.start_run(experiment_id=experiment_id, run_id=parent_run_id):
        with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
            child_run_id = child_run.info.run_id
            logger.info(
                f"Initiating new MLflow CHILD run '{child_run_name}' with run_id '{child_run_id}'"
            )
            mlflow.log_param("child_run_name", child_run_name)
            mlflow.log_param("its_parent_run_name", parent_run_name)

    run = mlflow.start_run(run_id=child_run_id)
    mlflow.autolog(log_models=False)
    mlflow.autolog(disable=True)

    return run


def load_last_parent_run(
    experiment_id: str,
    mlflow_experiment_run_name_mode: str,
    status: Optional[str] = None,
) -> Tuple[str, str]:

    # get parent name
    parent_run_name=get_name(run_mode=mlflow_experiment_run_name_mode, add_date=True)

    run = mlflow_run_finder(
        experiment_id=experiment_id,
        task_name=None,
        run_type="parent",
        status=status,
    )

    if run is None:
        return run, run
    run_id = run["run_id"]
    run_name = run["params.parent_run_name"]
    logger.info(f"Found last MLflow PARENT run {run_name} with run_id {run_id}")
    return run_id, run_name


def mlflow_run_finder(
    experiment_id: str,
    task_name: Optional[str] = None,
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    parent_run_name: Optional[str] = None,
) -> pd.Series:
    task_name_str = f"AND params.child_run_name = '{task_name}'" if task_name else ""
    status_str = f"AND status = '{status}'" if status else ""
    parent_str = (
        f"params.its_parent_run_name = '{parent_run_name}'"
        if parent_run_name is not None
        else ""
    )

    # usecase_str = f"AND tags.usecase = '{usecase}'" if usecase is not None else ""

    if run_type == "child":
        run_str = "params.child_run_name != '' "
    elif run_type == "parent":
        run_str = "params.parent_run_name != '' "
    else:
        run_str = ""

    # Formulate filter string
    filter_str = (
        run_str
        + parent_str
        + status_str
        + task_name_str
    )

    s = "\n\t" + filter_str.replace("AND", "\n\tAND")
    logger.info(f"Searching for MLflow run with criteria: {s}")

    if not experiment_id.isnumeric():
        # if we receive the experiment name, we need to get the experiment_id
        experiment_id = mlflow.get_experiment_by_name(experiment_id).experiment_id

    # Search for the run
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_str,
        max_results=10,
        order_by=["start_time DESC"],
    )
    if runs_df.empty:
        raise Exception("No MLflow run found")

    else:
        run_df = runs_df.iloc[0]
        logger.info(f" Found MLflow run with run_id {run_df['run_id']}")
        return run_df


def get_name(
    run_mode: Optional[str] = None,
    add_date: bool = False,
    truncate: bool = False,
) -> str:
    
    name = get_name_from_run_mode(run_mode)

    if add_date:
        today_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
        name = name + "_" + str(today_date)

    # For some operations like model serving, Databricks has a limit of 63 characters
    # for the model name. Thus we truncate the name if it is longer than 63 characters.
    # For other assets, it is not strictly necessary but this won't have any side effects.
    name = truncate_name(name) if truncate else name
    return name


def get_name_from_run_mode(s: str) -> str:
    try:
        if s.lower() == "username":
            res = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
            return res['tags']['user']
    except:
        return "run"


def log_to_mlflow(step: str, 
                  configuration: ConfigurationManager, 
                  model_obj: Optional[XGBClassifier] = None,
                  params: Optional[Dict] = None) -> None:
    
    if step == configuration.config.data_preprocessing.step_name:
        logger.info(f'logging started in mlflow for step {configuration.config.data_preprocessing.step_name}')
        for item in configuration.config.data_preprocessing:
            if 'data_path' in item:
                mlflow.log_param(item, configuration.config.data_preprocessing[item])
                if not 'input_data_path' in item:
                    table_timestamp = get_table_version(str(Path.cwd()) +'/'+configuration.config.data_preprocessing[item])
                    mlflow.log_param(item[:item.rindex('_')]+'_timestamp', table_timestamp)
                # mlflow.sklearn.log_model(
                #     sk_model=model_obj,
                #     artifact_path="model"
                # )
    elif step == configuration.config.model_training.step_name:
        logger.info(f'logging started in mlflow for step {configuration.config.model_training.step_name}')
        pip_dependencies = mlflow.xgboost.get_default_pip_requirements()
        # extra_pip_requirements = [
        #     wheel_path,
        # ]
        # pip_dependencies.extend(extra_pip_requirements)
        for item in configuration.config.model_training:
            mlflow.xgboost.log_model(
                xgb_model=model_obj,
                artifact_path="model",
                pip_requirements=pip_dependencies
            )
        boost_params = configuration.params.XGBClassifier
        mlflow.log_params({f"boost_{k}": v for k, v in boost_params.items()})
    elif step == configuration.config.model_evaluation.step_name:
        logger.info(f'logging started in mlflow for step {configuration.config.model_evaluation.step_name}')
        # for item in params.keys():
        mlflow.log_metrics(params['metrics'])
        mlflow.log_param('model_uri', params['model_uri'])


def load_dataset_from_run(run_df: pd.DataFrame, run_param_name: str):
    logger.info(f"Loading dataset '{run_param_name}' from run '{run_df['run_id']}'")

    path = run_df[f"params.{run_param_name}"]
    timestamp = run_df[f"params.{run_param_name[:run_param_name.rindex('_')]}_timestamp"]

    df = load_dataset(path, timestamp)

    mlflow.log_param(run_param_name, path)
    mlflow.log_param(run_param_name[:run_param_name.rindex('_')]+"_timestamp", timestamp)
    return df