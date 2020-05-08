# nba-salary-mvp
## Description
A linear neural network model to predict the salary and MVP vote share of NBA players from 1984-2017.

## Evaluating the model and example uses
Can be found on the following notebook   [![Binder](https://mybinder.org/badge_logo.svg)](https://hub.gke.mybinder.org/user/callumrai-nba-salary-mvp-e7kiyy6d/notebooks/results/Results%20Notebook.ipynb)

## Structure
* Data
  * Raw *Contains raw scraped data*
  * Clean *Contains scraped data cleaned to form features from*
  * Features *Contains features to train and predict on*
  * Predictions *Contains model final predictions*
  * *Final trained models and script to clean data*
* Features
  * *Creates features to train and predict on*
* Results
  * *Scripts to make prediction csvs, plots and notebook*
* Scrape
  * *Scrape all required data*
* Train
  * *Debugs parameter for and saves linear neural network*
