from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def get_message():
    # if request.method == "GET":
    print("Got request in main function")
    return render_template("home.html")


@app.route('/uploadText', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    return "56"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
