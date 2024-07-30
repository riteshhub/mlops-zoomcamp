"""Module to take care of creating a singleton of the execution environment class."""
import logging
import os
from re import S
from typing import Optional
from delta import configure_spark_with_delta_pip
from pyspark import SparkConf
from pyspark.sql import SparkSession

class PysparkSessionManager:
    from mlops import logger
    
    @classmethod
    def start_session(
        cls,
        app_name: Optional[str] = None,
        config: Optional[dict] = None,
        enable_hive_support: bool = False,
    ) -> SparkSession:
        """Get or create an execution environment session (currently Spark).

        It instantiates a singleton session that can be accessed anywhere from the
        lakehouse engine.

        Args:
            app_name: application name.
            config: extra spark configs to supply to the spark session.
            enable_hive_support: whether to enable hive support or not.
        """

        if "SIZENFIT_LOCAL_DEV_FLAG" in os.environ:
            spark_session = cls._start_session_local(
                app_name=app_name,
                config=config,
                enable_hive_support=enable_hive_support,
            )
        else:
            spark_session = cls._start_session_databricks(
                app_name=app_name,
                config=config,
                enable_hive_support=enable_hive_support,
            )

        return spark_session

    @classmethod
    def _start_session_databricks(
        cls,
        app_name: Optional[str] = None,
        config: Optional[dict] = None,
        enable_hive_support: bool = False,
    ) -> SparkSession:
        """Starts a spark session in databricks."""
        spark_session = SparkSession.getActiveSession()

        if not spark_session:
            default_config = {
                "spark.databricks.delta.optimizeWrite.enabled": True,
                "spark.sql.adaptive.enabled": True,
                "spark.databricks.delta.merge.enableLowShuffle": True,
            }
            cls.logger.info(
                f"Using the following default configs you may want to override them for "
                f"your job: {default_config}"
            )
            final_config: dict = {
                **default_config,
                **(config if config else {}),
            }
            cls.logger.info(f"Final config is: {final_config}")

            session_builder = SparkSession.builder.appName(app_name)
            if config:
                session_builder = session_builder.config(
                    conf=SparkConf().setAll(final_config.items())
                )
            if enable_hive_support:
                session_builder = session_builder.enableHiveSupport()
            spark_session = session_builder.getOrCreate()

            cls.logger.info("Pyspark session was started sucessfully")

        return spark_session

    @classmethod
    def _start_session_local(
        cls,
        app_name: Optional[str] = None,
        config: Optional[dict] = None,
        enable_hive_support: bool = False,
    ) -> SparkSession:
        """Starts a spark session locally."""

        try:
            del os.environ["SPARK_REMOTE"]
        except KeyError:
            pass

        builder = (SparkSession.builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )

        spark_session = configure_spark_with_delta_pip(builder).getOrCreate()

        return spark_session

    @classmethod
    def stop_session(self, spark_session: SparkSession) -> None:
        spark_session.stop()
        self.logger.info("Pyspark session was stoped sucessfully")
