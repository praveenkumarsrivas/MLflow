## Using the MLflow Client API
In this part of our journey we will be interfacing with the tracking server through one of the primary mechanism which we are going to use while training our ml model, i.e. `MlflowClient`.

#### Brief about `MlflowClient`:
This helps in creating and managing the experiments and runs and of a mLflow registry server that creates and manages the registered models and model versions

NOTE: For this experiment, this client will we our primary mlflow tracking capabilities and enables:
- Initiate a new experiment
- Start Runs with an experiment.
- Document parameter,metrics and tags for your run
- Log artifacts linked to runs, such as models, tables, plot and more.

```python
from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
```
These imports helps us to configure the client and specify about the  location of the trackingserver

## Configuring the MLflow Tracking Client
The default config of MLflow Tracking client local storage unless specified through the MLFLOW_TRACKING_URI environment variable.
therefore the experiments, data, models, and related attributes are stored within the active execution directory. However, to utilize a othe storage we need to specify the location from the tracking URI, the URI comprises the host, port component submitted as arguments to the mlflow server command, By specifying this to URI, one can connect to the designated tracking server and utilize it's functionalities instead of relying on local storage.
```python
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

```
Now we have a client interface to the tracking server that can both send data to and retrieve data from the tracking server

## The Default Experiment:
In MLflow the The Default Experiment refers to a container for organising and managing ML experiments, when you run a experiment using MLflow the results, metrics, parameters and it's artifacts are asociate with the experiments.
If we don't explicitly specify an experiment to use, MLflow automatically assign these results to the defaults experiments.

Here is the small example
```python
import mlflow

# Start MLflow run
with mlflow.start_run(): # default experiment
    # Log parameters
    mlflow.log_param("param1", 5)
    mlflow.log_param("param2", "value")

    # Train your model
    # (assume model training code here)

    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.1)

    # Save model
    # (assume model saving code here)
```
In the above scripts:
- We have started an MLflow experiment using `mlflow.start_run()`, which initiate the tracking for the current experiment.
- we log parameters such as `param1`, `param2` using `mlflow.log_param()`.
- we execute the model trainiing code.
- We log metrics such as `accuracy` and `loss` using `mlflow.metric()`
- we save trained model using example: `mlflow.sklearn.log_model()` if the model is scikit-learn model

## Searching Experiments
This function of mlflow helps to search the experiments when we have large number of experiments and we have to find/search the experiments from that.

### `MLflow` provides the two methods:
1. mlflow.search_run()
2. mlflow.search_experiments()

<br>

1. `mlflow.search_runs()`:

This method is used to search for individual runs within an experiment.
It allows you to filter and retrieve runs based on criteria such as `metric values, parameter values, tags, and more`.
It returns a DataFrame containing metadata about the matching runs.
Example usage:

```python
import mlflow

# Start an MLflow run and log some metrics and parameters
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_param("param2", "value")

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.1)

# Start another MLflow run with different metrics
with mlflow.start_run():
    mlflow.log_param("param1", 10)
    mlflow.log_param("param2", "another_value")

    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.08)

# Search for runs within the experiment with ID "0"
# Filter for runs where the accuracy metric is greater than 0.9
runs = mlflow.search_runs(experiment_ids="0", filter_string="metrics.accuracy > 0.9")

# Print the matching runs
print("Matching runs:")
for index, run in runs.iterrows():
    print("Run ID:", run.run_id)
    print("Parameters:", run.params)
    print("Metrics:", run.metrics)
    print("----------------------")


```
`2. mlflow.client.MlflowClient.search_experiments():`
- This method is used to search for experiments based on criteria such as name, artifact location, and tags.
- It allows you to retrieve metadata about experiments that match the specified criteria.
- It returns a list of Experiment objects containing metadata about the matching experiments.

Example usage:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

# Load Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Start MLflow experiment 1
mlflow.set_experiment("Hyperparameter Optimization for Random Forest Classifier")
with mlflow.start_run():

    # Define hyperparameters to tune
    n_estimators = 100
    max_depth = 5

    # Create and train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log hyperparameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    
# Start another MLflow experiment  
    
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

# Dummy customer churn data
data = {
    'age': [25, 35, 45, 55, 65],
    'income': [50000, 60000, 70000, 80000, 90000],
    'churn': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split features and target variable
X = df[['age', 'income']]
y = df['churn']

# Start MLflow experiment
mlflow.set_experiment("Logistic Regression for Customer Churn Prediction")
with mlflow.start_run():

    # Create and train Logistic Regression model
    clf = LogisticRegression()
    clf.fit(X, y)

    # Evaluate model
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)


# if we want to search experiment
experiments = client.search_experiments(filter_string="name LIKE '%Churn%'")  # searching the experiment name contains 'churn'

# Print information about the matching experiments
print("Matching experiments:")
for experiment in experiments:
    print("Experiment ID:", experiment.experiment_id)
    print("Experiment Name:", experiment.name)
    print("Artifact Location:", experiment.artifact_location)
    print("-----------------------------")


# we can get experiments by id
experiment_id = "296919399863335561"
# Search for experiments with the specified experiment ID
experiments = mlflow.get_experiment(experiment_id)
print(experiments)
```