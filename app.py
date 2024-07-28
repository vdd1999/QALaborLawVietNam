from flask import Flask
from flask import render_template, request, jsonify, make_response
from SupportFunction.MainFunction import get_answer, get_predicted_answer, get_combined_bm25_word2vec_scores
from SupportFunction.BM25Function import get_top_n_ranked_bm25

import uuid
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def main():
    return render_template('index.html', **globals())


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
    if data['type'] == 0:
        ans = get_top_n_ranked_bm25(data['message'], 5)
        print(ans)
        res_data = {
            'type': 0,
            'data': [{**ans, 'id': uuid.uuid4().hex} for ans in ans]
        }
        return make_response(jsonify(res_data), 200)
    elif data['type'] == 1:
        ques, ans = get_combined_bm25_word2vec_scores(data['message'])
        print(ques, ans)
        res_data = {
            'type': 1,
            'data': {
                'question': ques,
                'answer': ans
            }
        }
        return make_response(jsonify(res_data), 200)
    elif data['type'] == 2:
        ans = get_answer(data['message'])
        print(ans)
        res_data = {
            'type': 2,
            'message': ans.replace('</s>', '')
        }
        return make_response(jsonify(res_data), 200)
    elif data['type'] == 3:
        ans = get_predicted_answer(data['message'])
        print(ans)
        res_data = {
            'type': 3,
            'message': ans.replace('</s>', '')
        }
        return make_response(jsonify(res_data), 200)

    return make_response(jsonify({'message': 'Invalid request'}), 400)


if __name__ == "__main__":
    app.run()
