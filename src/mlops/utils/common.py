from datetime import datetime
import os
from box.exceptions import BoxValueError
from omegaconf import ListConfig, OmegaConf
from mlops import logger
import json
import xgboost
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from mlops.utils.spark_manager import PysparkSessionManager
from pyspark.sql import DataFrame
from typing import Optional


spark = PysparkSessionManager._start_session_local()

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = OmegaConf.load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    xgboost.dump(value=data, filename=path)
    logger.info(f"model file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = xgboost.load(path)
    logger.info(f"model file loaded from: {path}")
    return data

@ensure_annotations
def truncate_name(name: str) -> str:
    """Truncates a name to 63 characters, as Databricks has a limits of 63 chars
    on the length of the names of the different assets, especifically endpoint
    names and served models behind.

    Args:
        str: Non-truncated name.

    Returns:
        str: Truncated name.
    """

    MAX_LENGTH = 60  # As this is constant in Databricks, we leave it hardcoded
    if len(name) > MAX_LENGTH:
        logger.warning(f"Asset {name} is longer than {MAX_LENGTH} characters. Truncating it.")
        name = name[:MAX_LENGTH]
    return name


@ensure_annotations
def get_table_version(path: str) -> datetime:
    """Gets table timestampt version of a delta table

    Args:
        path (string): path to the table

    Returns:
        str: timestamo of the latest version of the delta table
    """

    # Get latest delta table version
    query = "DESCRIBE HISTORY delta.`{}`".format(path)
    table_version = (
        spark.sql(query).orderBy("version", ascending=False).limit(1).collect()[0]["timestamp"]
    )

    return table_version


@ensure_annotations
def load_dataset(
    path: str,
    timestamp: Optional[str] = None,
) -> DataFrame:
    
    logger.info(f"Loading '{path}' at timestamp '{timestamp}'")

    df = spark.read.format("delta").option("timestampAsOf", timestamp).load(path)

    return df


@ensure_annotations
def check_acceptance_criteria(acceptance_criteria: ListConfig, metric: dict) -> bool:
    status = True
    
    for item in acceptance_criteria:
        if item['condition'] == 'greater_than':
            if item['name'] in metric.keys():
                if metric[item['name']] < item['value']:
                    status = False
                    logger.info(f"{metric[item['name']]} < {item['value']}")
                    break
                else:
                    logger.info(f"{metric[item['name']]} > {item['value']}")
        if item['condition'] == 'less_than':
            if item['name'] in metric.keys():
                if metric[item['name']] > item['value']:
                    status = False
                    logger.info(f"{metric[item['name']]} > {item['value']}")
                    break
                else:
                    logger.info(f"{metric[item['name']]} < {item['value']}")

    return status