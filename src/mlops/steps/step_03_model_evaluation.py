from mlops.config.config_manager import ConfigurationManager
from mlops.components.model_evaluation import ModelEvaluation
from mlops.utils.mlflow_utils import (mlflow_run_manager, 
                                             mlflow_run_finder, 
                                             load_dataset_from_run, 
                                             log_to_mlflow)
from mlops.utils.common import check_acceptance_criteria
from mlops import logger
import mlflow
import sys



STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationStep:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        run = mlflow_run_manager(
                    mlflow_experiment_name=config_manager.config.mlflow_experiment_name,
                    mlflow_experiment_run_name_mode=config_manager.config.mlflow_experiment_run_name_mode,
                    child_run_name=config_manager.config.model_evaluation.step_name,
                    experiment_tracker_url=config_manager.config.mlflow_uri,
                    start=False,
        )
        experiment = mlflow.get_experiment_by_name(config_manager.config.mlflow_experiment_name)
        experiment_id = experiment.experiment_id
        parent_run_name = run.data.params["its_parent_run_name"]
        preprocess_run_df = mlflow_run_finder(
            experiment_id,
            "preprocess",
            run_type="child",
            parent_run_name=parent_run_name,
        )
        datasets = []

        if 'test_data_path' in config_manager.config.data_preprocessing.keys():
            train_data = load_dataset_from_run(run_df=preprocess_run_df, run_param_name='test_data_path')
            datasets.append(train_data)

        train_run_df = mlflow_run_finder(
            experiment_id,
            "train",
            run_type="child",
            parent_run_name=parent_run_name,
        )

        artifact_uri = train_run_df["artifact_uri"]
        model_uri = artifact_uri + "/model"

        logger.info(f"Loading model from train run_id: {train_run_df['run_id']}")
        model = mlflow.xgboost.load_model(model_uri)

        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        eval_report = model_evaluation.evaluate(datasets=datasets, model=model)
        param = {'metrics': eval_report, 'model_uri': model_uri}
        log_to_mlflow(step=model_evaluation_config.step_name, configuration=config_manager, model_obj=model, params=param)

        logger.info("############### Computing acceptance criteria")
        acceptance_criteria = model_evaluation_config.acceptance_criteria
        model_accepted = check_acceptance_criteria(acceptance_criteria, eval_report)
        if not model_accepted:
            logger.info("Terminating process as Model is unable to meet defined criteria")
            mlflow.end_run()
            sys.exit(1)

        mlflow.end_run()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationStep()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
