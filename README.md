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




# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 324037277821.dkr.ecr.eu-north-1.amazonaws.com/mlflows
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one
     

# To get an self hosted on online
	   
### Stop the self-hosted runner application if it is currently running.

### Install the service with the following command:

    sudo ./svc.sh install
### Alternatively, the command takes an optional user argument to install the service as a different user.

    ./svc.sh install USERNAME

## Starting the service

### Start the service with the following command:

    sudo ./svc.sh start

## Checking the status of the service
### Check the status of the service with the following command:

    sudo ./svc.sh status
    
	### For more information on viewing the status of your self-hosted runner, see "Monitoring and troubleshooting self-hosted runners."

## Stopping the service
### Stop the service with the following command:

     sudo ./svc.sh stop


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = eu-north-1

    AWS_ECR_LOGIN_URI = demo>>  324037277821.dkr.ecr.eu-north-1.amazonaws.com

    ECR_REPOSITORY_NAME = mlflows




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model

 