from flask import Flask, request, redirect, flash, abort
from flask_cors import CORS
from flask_restful import Api
import os
import json

# Model Processors
import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import *
from custom_dataset import *

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

app.secret_key = b'secret'

BASE_PATH = os.path.abspath("./")

X_test = pd.read_csv("final_samples_preprocessed.csv")
original_df = pd.read_csv("final_samples_original.csv")

model_name = f'5_layer_256_BN_dropout'
model = nn_Regression(input_features = 299, dropout= 0.5, model_name=model_name)
checkpoint = torch.load("./checkpt_5_layer_256_BN_dropout_470.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

def test_model(model, testloader, device='cpu'):
    model.to(device)
    model.eval()

    for tweets, labels in testloader:
        tweets, labels = tweets.float().to(device), labels.float().to(device)
        output = model.forward(tweets)
        return output

def extract_sample_tweets(sample_no):
    sample_tweets = original_df.sample(sample_no)
    sample_tweets["tweet_id"] = sample_tweets["tweet_id"].astype(str)
    return sample_tweets

@app.route('/tweets', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.get_json():
            tweet_to_predict = request.get_json()
            y_test = [tweet_to_predict["retweets"],]
            tweet_df_to_predict = X_test.loc[(X_test["tweet_id"] == int(tweet_to_predict["tweet_id"]))]
            test_dataset = Custom_Testing_Dataset(tweet_df_to_predict.drop(columns=['tweet_id']),y_test)
            test_loader = DataLoader(test_dataset)
            result = test_model(model, test_loader)
            return str(round(result.item()))
        else:
            flash('No data')
            return redirect(request.url)
    elif request.method == "GET":
        return json.dumps(extract_sample_tweets(4).to_dict("records"))
    else:
        return abort(404)


if __name__ == '__main__':
    app.run(debug=True)
