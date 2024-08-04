# DistilBERT-based Sentiment Analysis Project for Predicting Shoe Review Ratings

This project implements a sentiment analysis model to predict star ratings for Amazon shoe reviews. It leverages DistilBERT-base-uncased, a pre-trained transformer model from Hugging Face, fine-tuned on a dataset of Amazon shoe reviews.

## Project Structure

- `01. Data Preparation.ipynb`: This notebook handles the entire data pipeline:
  * __Data Collection:__ An amazon-shoe-review dataset has been collected from [here](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Shoes_v1_00.tsv).
  * __Data Cleaning & Preprocessing:__ Data cleaning and preprocessing has been done to prepare it for model training.
  * __Data Sharing:__ After preprocessing the dataset has been pushed to HuggingFace Hub. [Dataset Link](https://huggingface.co/datasets/mazed/amazon_shoe_review)

- `02. Model Training.ipynb`: This notebook covers:
  * Fine-tuning the pre-trained DistilBERT-base-uncased model from Hugging Face on the preprocessed data for predicting shoe review star ratings.

- `03. Save Model to Hub.ipynb`: This notebook handles:
  * __Model Evaluation:__ Predicitons are made on few examples to evaluate the fine-tuned model.
  * __Model Sharing:__ The fine-tuned model is then pushed to HuggingFace model hub. [Model Link](https://huggingface.co/mazed/distilbert-amazon-shoe-review)

- `requirements.txt`: Lists the dependencies needed for the project:
  - `transformers`
  - `gradio`
  - `torch`

- `app.py`: A script to deploy the model using Gradio for a web-based interface.

