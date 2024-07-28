from SupportFunction.BM25Function import get_top_n_ranked_bm25

answers = get_top_n_ranked_bm25(input("Nhập câu hỏi: "), 5)

for ans in answers:
    print(ans['question'])
    print(ans['score'])
    print()
