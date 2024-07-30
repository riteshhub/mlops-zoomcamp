# mlops-zoomcamp

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the step configurations in src config
5. Update the configuration manager in src config
6. Update the components
7. Update the steps 
8. Update the pipeline.py

## Prerequsite

To run in local, follow https://sparkbyexamples.com/pyspark-tutorial/ link to setup environment


## How to run?
#### STEP 01- Clone the repository
```bash
https://github.com/riteshhub/mlops-zoomcamp
```
#### STEP 02- Create a conda environment after opening the repository
```bash
conda create -n <<env_name>> python=3.11 -y
```

```bash
conda activate <<env_name>>
```


#### STEP 03- install the requirements
```bash
pip install -r requirements.txt
```

#### STEP 04- start mlflow server
```bash
mlflow server --host localhost --port 5000
```

#### STEP 05- trigger pipeline
```bash
python pipeline.py
```