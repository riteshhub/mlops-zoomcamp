from mlops.steps.step_01_data_preprocessing import DataPreProcessingStep
from mlops.steps.step_02_model_training import ModelTrainingStep
from mlops.steps.step_03_model_evaluation import ModelEvaluationStep
from mlops.steps.step_04_model_registry import ModelRegistryStep
from mlops.steps.step_05_model_deployment import ModelDeploymentStep
from mlops import logger


STAGE_NAME = "Data PreProcessing stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_preprocessing = DataPreProcessingStep()
   data_preprocessing.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
   model_training = ModelTrainingStep()
   model_training.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationStep()
   model_evaluation.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Registry stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = ModelRegistryStep()
   obj.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Deployment stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = ModelDeploymentStep()
   obj.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e