import json
import os
import sys
import pickle
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
    # get the values from website
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

    # define the column names for the dataframe
    columns = ["@ in URLs", "Attachments",
               "Css", "Encoding", "External Resources", "Flash content", "HTML content", "HTML Form",
               "HTML iFrame", "IPs in URLs", "Javascript", "URLs"]
    #print("Columns Set")

    dfWebsite = pd.DataFrame({"id": [0]})
    for value, colum in zip(valuesList, columns):
        dfWebsite[colum] = int(value)

    df = dfWebsite.drop(columns=["id"])

    #open and save the pickle models
    with open("./pickleModels/LGmodel", "rb") as f:
        LGmodel = pickle.load(f)
    with open("./pickleModels/KNNmodel", "rb") as f:
        KNNmodel = pickle.load(f)
    with open("./pickleModels/NBmodel", "rb") as f:
        NBmodel = pickle.load(f)

    LGpred = LGmodel.predict(df.values)
    KNNpred = KNNmodel.predict(df.values)
    NBpred = NBmodel.predict(df.values)

    # get the dataManager and a clean df to make a train test split to then get extra values
    import sys
    sys.path.append("./dataScience")
    from dataScience import dataManager as dm
    from dataScience import graphManager as gm
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()

    cleanDF = dm.getCleanData()

    X = cleanDF.drop("Phishy", axis=1)
    y = cleanDF["Phishy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create mat 
    matLG = gm.make7x7ConsusionMatrix(y_test, LGpred, "Logistic Regression", "Predicted", "Actual"
    "./images/LG_ConfusionMatrixWebsiteValues.png")

    print(matLG)
    

    return json.dumps({
        "value": 1,
        "value2": 2
    })

# @app.route("/", methods=['GET'])
# def index():
#    print("index called")
#    return "index"


app.run(host='0.0.0.0', port=3001)
