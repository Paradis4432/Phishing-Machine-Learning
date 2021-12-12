import json, os, sys
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

import test as t

@app.route('/check', methods=['POST'])
def check():
    #x = t.multiply(3,5)
    #print(x)
    #print("check called")
    #print("num:", num["value"])
    #print("num2:", num["value2"])
    #multi = t.multiply(num["value"], num["value2"])
    
    #return json.dumps([num["value"], num["value2"]])

    values = request.get_json()
    ArrinUrls = values["ArrinUrls"]
    Attachments = values["Attachments"]
    Css = values["Css"]
    Encoding = values["Encoding"]
    External_Resources = values["External_Resources"]
    Flash_content = values["Flash_content"]
    HTML_content = values["HTML_content"]
    HTML_Form = values["HTML_Form"]
    HTML_iFrame = values["HTML_iFrame"]
    IPsInURLs = values["IPsInURLs"]
    Javascript = values["Javascript"]
    URLs = values["URLs"]

    print(values)

    return json.dumps({
        "value" : 1,
        "value2" : 2
    })

#@app.route("/", methods=['GET'])
#def index():
#    print("index called")
#    return "index"


app.run(host='0.0.0.0', port=3001)