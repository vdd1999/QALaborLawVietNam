from flask import Flask
from flask import render_template, request, url_for, jsonify, make_response
from SupportFunction.BM25Function import find_best_matching_question, get_top_n_ranked_bm25
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def main():
    return render_template('index.html', **globals())


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
    if data['type'] == 0:
        ques, ans = find_best_matching_question(data['message'])
        res_data = {
            'type': 0,
            'message': ques,
            'answer': ans
        }
        return make_response(jsonify(res_data), 200)
    elif data['type'] == 1:
        data = get_top_n_ranked_bm25(data['message'])
        res_data = {
            'type': 1,
            'data': data
        }
        return make_response(jsonify(res_data), 200)

    return make_response(jsonify({'message': 'Invalid request'}), 400)
