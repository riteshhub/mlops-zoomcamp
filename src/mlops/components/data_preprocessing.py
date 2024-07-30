import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlops import logger
from mlops.config.config_steps import DataPreProcessingConfig
from mlops.utils.spark_manager import PysparkSessionManager
import mlflow

spark = PysparkSessionManager._start_session_local()

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        """initilize class

        Args:
            config (DataPreProcessingConfig): object
        """
        self.config = config


    def perform_preprocessing(self, df:pd.DataFrame) -> ColumnTransformer:
        """encapsulate logic to perform pre prpocessing on input data

        Raises:
            raises exception with error message upon failure
        
        Returns:
            None
        """
        try:
            logger.info('pre processing started')
            # logger.info(f"FEATURES {self.config.all_schema.FEATURES}")
            # num_cols = df[self.config.all_schema.FEATURES].select_dtypes(include=['float64','int64']).columns
            # cat_cols = df[self.config.all_schema.FEATURES].select_dtypes(include='object').columns
            num_cols = list(self.config.all_schema.NUMERICAL_FEATURES.keys())
            cat_cols = list(self.config.all_schema.CATEGORICAL_FEATURES.keys())

            logger.info(f"numerical columns {num_cols}")
            logger.info(f"categorical columns {cat_cols}")

            num_pipe = Pipeline(steps=[
                ('num_cols_impute', SimpleImputer(strategy='mean')),
                ('num_cols_scale', StandardScaler())
            ])

            cat_pipe = Pipeline(steps=[
                ('cat_cols_impute', SimpleImputer(strategy='most_frequent')),
                ('cat_cols_encode', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocess = ColumnTransformer(transformers=[
                ('num_col_trans', num_pipe, num_cols),
                ('cat_col_trans', cat_pipe, cat_cols)
            ], remainder='passthrough')

            logger.info('pre processing completed')
            
            transformer = preprocess.fit(df)
            mlflow.sklearn.log_model(
                        sk_model=transformer,
                        artifact_path="sk_models"
                )
            return transformer
        
        except Exception as e:
            raise e
        
    def write_processed_data(self, data: np.ndarray, train_validation_test_split:str)-> None:
        
        np.random.shuffle(data)

        train_ratio = float(train_validation_test_split.split(':')[0])
        validation_ratio = float(train_validation_test_split.split(':')[1])

        train, validation, test = np.split(data, [int(train_ratio * len(data)), int((train_ratio+validation_ratio) * len(data))])
        
        logger.info("Saving training dataset to {}".format(self.config.train_data_path))
        train_df = spark.createDataFrame(train)
        train_df.write.format("delta").mode("overwrite").options(path=self.config.train_data_path, overwriteSchema="True").save()

        logger.info("Saving validation dataset to {}".format(self.config.validation_data_path))
        validation_df = spark.createDataFrame(validation)
        validation_df.write.format("delta").mode("overwrite").options(path=self.config.validation_data_path, overwriteSchema="True").save()

        logger.info("Saving test dataset to {}".format(self.config.test_data_path))
        test_df = spark.createDataFrame(test)
        test_df.write.format("delta").mode("overwrite").options(path=self.config.test_data_path, overwriteSchema="True").save()