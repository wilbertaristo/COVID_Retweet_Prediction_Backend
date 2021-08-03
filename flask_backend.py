from flask import Flask, request, redirect, flash, abort,  Response
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
from utils import *

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

app.secret_key = b'secret'

BASE_PATH = os.path.abspath("./")

X_test = pd.read_csv("test_samples_preprocessed.csv", index_col=0)
original_df = pd.read_csv("test_samples_original.csv", index_col=0)

def load_model(path):
    saved_model = torch.load(path)
    return saved_model

model = load_model("./4_layer_BN_dropout50_50.pt")

def test_model(model, testloader, device='cuda'):
    model.to(device)
    model.eval()

    for tweets, labels in testloader:
        tweets, labels = tweets.to(device), labels.to(device)

        output = model.forward(tweets.float())
        return output

def extract_sample_tweets(sample_no):
    df = X_test.sample(sample_no)
    tweet_ids = df['tweet_id'].tolist()
    sample_tweets = pd.DataFrame(columns=['tweet_id','username','timestamp','followers','friends',
                                                  'retweets','favourites','entities','sentiments','mentions',
                                                  'hashtags','url']) 
    for tweet_id in tweet_ids:
        sample_tweets = sample_tweets.append(original_df[(original_df['tweet_id']==tweet_id)])
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
            print(round(result.item()))
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
