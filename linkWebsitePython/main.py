import json, os, sys, pickle
import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

#import test as t
#import v1.ipynb as v1
#from ipynb.fs.defs.v1 import test as te

@app.route('/check', methods=['POST'])
def check():
    #x = t.multiply(3,5)
    #print(x)
    #print("check called")
    #print("num:", num["value"])
    #print("num2:", num["value2"])
    #multi = t.multiply(num["value"], num["value2"])
    
    #return json.dumps([num["value"], num["value2"]])

    #print("Starting check")

    valuesList = []
    values = request.get_json()
    valuesList.append(values["ArrinUrls"])
    valuesList.append(values["Attachments"])
    valuesList.append(values["Css"])
    valuesList.append(values["Encoding"])
    valuesList.append(values["External_Resources"])
    valuesList.append(values["Flash_content"])
    valuesList.append(values["HTML_content"])
    valuesList.append(values["HTML_Form"])
    valuesList.append(values["HTML_iFrame"])
    valuesList.append(values["IPsInURLs"])
    valuesList.append(values["Javascript"])
    valuesList.append(values["URLs"])
    #print("Values Set")

    columns=["@ in URLs", "Attachments", 
    "Css", "Encoding", "External Resources", "Flash content", "HTML content", "HTML Form",
    "HTML iFrame", "IPs in URLs", "Javascript", "URLs"]
    #print("Columns Set")
    
    df = pd.DataFrame({"id": [0]})
    for value, colum in zip(valuesList, columns):
        #print("Value:", value)
        #print the type of the value
        #print("Type:", type(value))
        #print("Column:", colum)
        df[colum] = int(value)
    df = df.drop(columns=["id"])

    #print("DataFrame:", df)
    print(os.getcwd())

    with open("./linkWebsitePython/modelLG_pkl", "rb") as f:
        lgModel = pickle.load(f)
    with open("./linkWebsitePython/modelKNN_pkl", "rb") as f:
        knnModel = pickle.load(f)
    with open("./linkWebsitePython/modelNB_pkl", "rb") as f:
        nbModel = pickle.load(f)

    LGpred = lgModel.predict(df.values)
    KNNpred = knnModel.predict(df.values)
    NBpred = nbModel.predict(df.values)
    print("NBpred:", LGpred)
    print("NBpred:", KNNpred)
    print("NBpred:", NBpred)


    return json.dumps({
        "value" : 1,
        "value2" : 2
    })

#@app.route("/", methods=['GET'])
#def index():
#    print("index called")
#    return "index"


app.run(host='0.0.0.0', port=3001)