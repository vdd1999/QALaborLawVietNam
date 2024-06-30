from flask import Flask
from flask import render_template, request, url_for, jsonify, make_response

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def main():
    return render_template('index.html', **globals())


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
