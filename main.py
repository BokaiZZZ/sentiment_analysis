from flask import Flask, jsonify
from predict import predict_label
import nltk

app = Flask(__name__)


@app.route("/")
def hello():
    """Return a friendly HTTP greeting."""
    return "Welcome to this sentimantal trend analysis interface!"


@app.route("/predict/<text>")
def predict(text):
    pred = predict_label(text)
    val = {"text":text, "trend": pred}
    return jsonify(val)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
