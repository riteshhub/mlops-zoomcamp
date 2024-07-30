from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel

class LoanDetails(BaseModel):
    Gender: str
    Married: str
    Dependents: float
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@dataclass(frozen=True)
class DataPreProcessingConfig:
    """Declare data pre processing related configs
    """
    step_name: str
    root_dir: Path
    input_data_path: Path
    train_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    train_validation_test_split: str
    all_schema: dict

@dataclass(frozen=True)
class ModelTrainingConfig:
    """Declare model training related configs
    """
    step_name: str
    root_dir: Path
    model_name: str
    n_estimators: int
    max_depth: int
    learning_rate: int
    objective: str
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Declare model evaluation related configs
    """
    step_name: str
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    acceptance_criteria: dict

@dataclass(frozen=True)
class ModelRegistryConfig:
    """Declare model registry related configs
    """
    step_name: str
    register_model_name: str

@dataclass(frozen=True)
class ModelDeploymentConfig:
    """Declare model deployment related configs
    """
    step_name: str
    register_model_name: str

@dataclass(frozen=True)
class ModelObservabilityConfig:
    """Declare model observability related configs
    """
    project_id: str
    dashboard_name: str
    root_dir: Path
    input_data_path: Path
    report_path: Path
    all_schema: dict