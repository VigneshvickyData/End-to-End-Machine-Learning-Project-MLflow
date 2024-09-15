# End-to-End-Machine-Learning-Project-MLflow



## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py


# dagshub
## How to run?
### STEPS:

Clone the repository

```bash
https://github.com/VigneshvickyData/End-to-End-Machine-Learning-Project-MLflow
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='candycrushvicky7', repo_name='End-to-End-Machine-Learning-Project-MLflow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/candycrushvicky7/End-to-End-Machine-Learning-Project-MLflow.mlflow

export MLFLOW_TRACKING_USERNAME=candycrushvicky7

export MLFLOW_TRACKING_PASSWORD=e72434b23a653d24e6bc9ed73def97b5e7cc0fa9

```
