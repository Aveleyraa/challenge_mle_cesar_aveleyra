# Documentation Flight Delay Prediction API

## Disclaimer

Due to time constraints, the API was not fully deployed to the cloud.  
However, the complete deployment process is fully documented in this repository, including the infrastructure design, CI/CD pipelines, and the steps required to deploy the API on AWS.



## Project Overview

### Objective
Develop a flight delay prediction API using Machine Learning, deployed to the cloud with automated CI/CD.

### Technology Stack
- **ML**: XGBoost, Scikit-learn
- **API**: FastAPI, Uvicorn
- **Cloud**: AWS *
- **CI/CD**: GitHub Actions
- **Language**: Python 3.11

### Main Features
Delay prediction with balanced XGBoost   
Automatic cloud deployment  
Automated testing (CI)  
Continuous deployment (CD)  

---

## IMPORTANT

### `pyproject.toml` instead of `requirements.txt`

In this project, I decided to use **`pyproject.toml` instead of a traditional `requirements.txt`** in order to follow modern Python packaging and MLOps best practices.

The workflow starts by creating an isolated virtual environment to manage dependencies in a clean and reproducible way. For this, I used **`uv`** as the environment and dependency manager:

```bash
uv venv mlops-env
source mlops-env/bin/activate
```
Once the virtual environment is activated, the project dependencies are defined inside a pyproject.toml file. This allows all dependencies to be structured and centralized in a single configuration file, including:

Core dependencies ([project.dependencies])

Development dependencies ([project.optional-dependencies.dev])

Tooling configuration (formatters, linters, test frameworks)

Dependencies are installed using:

```bash
uv pip install -e ".[dev]"
```
## Project Structure

```
.
├── challenge
│   ├── api.py
│   ├── exploration.ipynb
│   ├── exploration.py
│   ├── __init__.py
│   ├── model.py
├── data
│   └── data.csv
├── Dockerfile
├── docs
│   └── challenge.md
├── infra
│   ├── cloudformation.yml
│   ├── parameters_dev.json
│   └── parameters_prod.json
├── __MACOSX
│   ├── challenge
│   ├── docs
│   ├── tests
│   │   ├── api
│   │   ├── model
│   │   └── stress
│   └── workflows
├── Makefile
├── pyproject.toml
├── README.md
├── reports
│   ├── html
│   └── junit.xml
├── requirements-dev.txt
├── requirements-test.txt
├── requirements.txt
├── tests
│   ├── api
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── test_api.py
│   ├── __init__.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── test_model.py
│   ├── __pycache__
│   └── stress
│       ├── api_stress.py
│       └── __init__.py
├── uv.lock
└── workflows
    ├── cd.yml
    └── ci.yml
```


## Part I: Machine Learning Model

### 1.1 Problem Analysis

**Context:**
- Flight dataset with operational features
- Target: `delay` (1 if delay > 15 min, 0 otherwise)
- **Class imbalance**: ~81% class 0, ~19% class 1

**Initial Metrics:**

**XGBoost without balancing:**
```
              precision    recall  f1-score   support
           0       0.81      1.00      0.90     18294
           1       0.00      0.00      0.00      4214
```
Problem: Predicts everything as class 0

**Logistic Regression without balancing:**
```
              precision    recall  f1-score   support
           0       0.82      0.99      0.90     18294
           1       0.56      0.03      0.06      4214
```
Problem: Very low recall for class 1

### 1.2 Implemented Solution

Now, for models using the top 10 features of importance and that are balanced, the following results are obtained:

**Top 10 Features:**
1. OPERA_Latin American Wings
2. MES_7 (July)
3. MES_10 (October)
4. OPERA_Grupo LATAM
5. MES_12 (December)
6. TIPOVUELO_I (International)
7. MES_4 (April)
8. MES_11 (November)
9. OPERA_Sky Airline
10. OPERA_Copa Air

Logistic Regression balancing:

```
Matrix: [[9487, 8807], [1314, 2900]]
              precision    recall  f1-score   support

           0       0.88      0.52      0.65     18294
           1       0.25      0.69      0.36      4214


```

XGBoost balancing:
```
Matrix: [[9556, 8738], [1313, 2901]]
              precision    recall  f1-score   support

           0       0.88      0.52      0.66     18294
           1       0.25      0.69      0.37      4214

```

They are practically the same, and being unbalanced, they don't improve anything. Therefore, taking this into consideration, I would use XGBoost since it has advantages such as being able to adjust more hyperparameters when trying to improve it, and in general, it is more robust.

**Final Model: XGBoost with class balancing**

### Class creation for the model.py file

A standalone .py file was generated from the original notebook using pytext, as a best practice to improve code maintainability and facilitate the transition from experimentation to production-ready code.

Based on this file, the core methods provided by the template were implemented: preprocess, fit, and predict, reusing the logic developed during the exploratory phase in the notebook.

In addition, several helper functions were created to support the preprocessing pipeline, including:

- _generate_target: to compute the target variable.

- _get_min_diff: to calculate time differences in minutes.

- predict_proba: to return the probability of flight delay instead of only the binary prediction.




---

## Part II: FastAPI Implementation

### 2.1 API Structure

**File: `challenge/api.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

app = FastAPI()
model = DelayModel()


```

### 2.2 Endpoints

#### GET /health
**Description:** API health check

**Response:**
```json
{
  "status": "OK"
}
```

#### POST /predict
**Description:** Flight delay prediction

**Request:**
```json
{
  "flights": [
    {
      "OPERA": "Grupo LATAM",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

**Response:**
```json
{
  "predict": [0]
}
```

**Validations:**
- `MES`: 1-12
- `TIPOVUELO`: 'N' or 'I'
- `OPERA`: Valid airline



### 2.3 Error Conversion 422 → 400

**Problem:** FastAPI returns 422 by default, but tests expect 400.

**Solution:** Custom exception handler

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=400,  # ← Convert to 400
        content={"detail": exc.errors()[0]["msg"]}
    )
```



### 2.4 Run API Locally

```bash
# With uvicorn
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000


# Verify
curl http://localhost:8000/health
```

---

## Part III: Cloud Deployment

##  Requirements

- **AWS CLI** configured with permissions for  S3, IAM, CloudFormation, lambda and API gateway
- **jq** (for JSON parameter handling in infra deploys)

##  3.1 Connect GitHub Actions to AWS using OIDC

1. Create an IAM Identity Provider in your AWS account for GitHub OIDC. [AWS link example](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html)

2. Create an IAM Role in your AWS account with a trust policy that allows GitHub Actions to assume it:
```bash
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:<GITHUB_ORG>/<GITHUB_REPOSITORY>:ref:refs/heads/<GITHUB_BRANCH>"
        }
      }
    }
  ]
}
```
3. Attach permissions to the IAM Role that allow it to access the AWS resources you need.

4. Create GitHub Actions workflow (in this repo there are 4 differents fo different purposes):

##  Add GitHub Secrets & Variables

In repository:

**Settings → Secrets and variables → Actions**

### Secrets (sensitive)
- `AWS_ROLE_TO_ASSUME` → your OIDC role ARN  
  _Example:_ `arn:aws:iam::123456789012:role/github-oidc-actions`


### Variables (non-sensitive defaults; you can override per workflow)
- `AWS_ACCOUNT_ID` → `123456789012`
- `AWS_DEFAULT_REGION` → `us-east-2`
- `COMPANY_NAME` → `latam`
- `PROJECT_NAME` → `delay-prediction`
- `ENV` → `dev` _(or `prod`)_

---


##  Infrastructure Deployment


1. Go to **GitHub → Actions → Infra.yml**.
2. Click **Run workflow** (top-right).
3. Choose the input **`env`** (e.g., `dev` or `prod`).
4. Click **Run workflow** and wait for the job to finish.

**What this workflow does**
1. Assumes your AWS role via OIDC (using `AWS_ROLE_TO_ASSUME`).
2. Validates the CloudFormation template (`infra/cloudformation.yml`).
3. Loads parameters from `infra/parameters_<env>.json`.
4. Creates/updates the CloudFormation stack `latam-delay-infra-<env>`.
5. Prints the stack **Outputs** (e.g., bucket names, ARNs) at the end of the job.

```bash
      - name: Deploy infrastructure
        run: |
          set -e
          echo "🚀 Deploying CloudFormation stack"
          aws cloudformation deploy \
            --template-file infra/cloudformation.yml \
            --stack-name $STACK_NAME \
            --capabilities CAPABILITY_NAMED_IAM \
            --parameter-overrides $PARAMS \
            --no-fail-on-empty-changeset
```

### AWS Deployment Architecture (S3 + Lambda + API Gateway)

The objective of deploying on AWS was to expose the Machine Learning model as a serverless service, accessible via HTTP, **Unfortunately, I couldn't finish the implementation on AWS, however, I've documented how it should work here**



The proposed architecture is based on the following services:

```
Client (curl / frontend / locust)
        |
        v
API Gateway (public endpoint)
        |
        v
AWS Lambda (FastAPI + model)
        |
        v
ML Model loaded in memory
        |
        v
JSON response with prediction
```
### Step by Step:

1. **Client sends request**
   ```bash
   curl -X POST https://6fnfe0p79e.execute-api.us-east-2.amazonaws.com/dev\
     -H "Content-Type: application/json" \
     -d '{"flights": [{"OPERA": "LATAM", "TIPOVUELO": "I", "MES": 7}]}'
   ```

2. **API Gateway receives and validates**
   - Verifies headers
   - Applies rate limiting (if configured)
   - Handles CORS
   - Invokes Lambda

3. **Lambda initializes** 
   - Loads code
   - Loads model into memory
   - Optionally downloads from S3


4. **FastAPI processes**
   - Calls DelayModel
   - Generates prediction

5. **Lambda responds**
   - Returns to API Gateway

6. **API Gateway responds to client**
   ```json
   {
     "predict": [0]
   }
   ```

---



## Part IV: Continuous Integration (ci.yml)

This file defines the Continuous Integration pipeline.  
Its main goal is to automatically validate the code every time changes are made.

### When does it run?

- On every push to the `develop` branch.  
- On every Pull Request targeting `main` or `develop`.

### What does it do?

1. Checks out the repository code.  
2. Sets up Python 3.11.  
3. Installs the project dependencies (`.[dev]`).  
4. Runs the model tests.  
5. Runs the API tests.  
6. Generates test and coverage reports.  
7. Uploads the reports as workflow artifacts.

This pipeline ensures that both the model and the API work correctly before allowing any deployment. If the tests fail, the CD pipeline should not run.

---

##  Continuous Deployment (cd.yml)

This file defines the Continuous Deployment pipeline.  
Its goal is to automatically deploy the application to AWS once the code has passed all validations.

### When does it run?

- Automatically when the CI workflow finishes successfully.  
- Only if the commit comes from the `develop` branch.  
- It can also be triggered manually (`workflow_dispatch`).

### What does it do?

1. Checks out the repository code.  
2. Authenticates with AWS using OIDC.  
3. Installs the project dependencies.  
4. Packages the API code into a `lambda.zip` file.  
5. Updates the AWS Lambda function code.  
6. Publishes a new Lambda version (restarts the service).

This pipeline takes the code validated by CI and automatically deploys it as an API on AWS Lambda.










