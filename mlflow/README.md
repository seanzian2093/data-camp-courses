# Introduction to MLflow

## Intro

* Start a run
* Logging a run
* End a run
  * Only when a run is stopped, artifacts are flushed
  * Better using a with block
* Querying runs

## MLflow Model

* Introduction to MLflow model
* Model API
* Custom model
* Saving and loading a model
  * Saving to a file system location so model can be copied/loaded
  * A saved model is Not necessarily tracked
* Logging a model
  * Logging a model together with its param/metrics/input expamle/signature, etc for tracking purpose. E.g., picking a best model from various parameter combinations
  * model is saved under run id

* Serving a model
  * A RESTful API
  * File system
  * Run ID with tracking name
  * AWS S3 bucket

## MLFlow Model Registry

* Registering a model
A best model from development that is ready for testing, pushing to production ultimately
* Transitioning a model
  * Pushing model from lower environemnt to higher, ultimately production
  * Retiring a model, i.e., archiving
* Three environment
  * Staging
  * Production
  * Archived

## MLFlow Projects

* `mlflow.projects`
* MLFlow CLI
* Parameters
* Workflow
