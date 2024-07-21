from flask import Flask
from flask import render_template, request, url_for, jsonify, make_response
from SupportFunction.MainFunction import get_answer, get_predicted_answer
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def main():
    return render_template('index.html', **globals())


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
    if data['type'] == 0:
        ans = get_answer(data['message'])
        res_data = {
            'type': 0,
            'message': ans
        }
        return make_response(jsonify(res_data), 200)
    elif data['type'] == 1:
        ans = get_predicted_answer(data['message'])
        res_data = {
            'type': 1,
            'message': ans
        }
        return make_response(jsonify(res_data), 200)

    return make_response(jsonify({'message': 'Invalid request'}), 400)
