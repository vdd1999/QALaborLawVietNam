# WORD 2 VEC
from gensim.models import Word2Vec

# Huấn luyện mô hình Word2Vec với các đoạn văn bản đã được token hóa
#Bước 1 Token data để pre train model
split_context_pre_train = [context.split() for context in tokenized_contexts]

# Định nghĩa model và khai báo các biến
# Trong đó
# vector_size: Đây là kích thước của vector ngữ nghĩa, tức là số chiều của không gian vector mà từng từ sẽ được biểu diễn. Một giá trị lớn hơn cho phép mô hình lưu trữ nhiều thông tin hơn nhưng cũng đòi hỏi nhiều dữ liệu hơn để huấn luyện và có thể làm tăng thời gian tính toán.

# window: Kích thước của "cửa sổ ngữ cảnh" mà mô hình sẽ xem xét xung quanh một từ. Nói cách khác, đây là số lượng từ trước và sau từ hiện tại mà mô hình sẽ sử dụng để học ngữ cảnh của từ đó. Giá trị window lớn hơn cho phép mô hình học được nhiều ngữ cảnh xa hơn nhưng cũng tăng độ phức tạp tính toán.

# min_count: Số lần tối thiểu một từ phải xuất hiện trong tập dữ liệu để được xem xét trong mô hình. Các từ xuất hiện ít hơn min_count lần sẽ bị bỏ qua. Điều này giúp loại bỏ các từ hiếm và giảm kích thước của mô hình.

# workers: Số lượng luồng sử dụng để huấn luyện mô hình. Tăng số lượng luồng có thể giúp tăng tốc độ huấn luyện nhưng cũng phụ thuộc vào số lõi CPU có sẵn.

# epochs: Số lần lặp qua toàn bộ tập dữ liệu trong quá trình huấn luyện. Nhiều epochs có thể cải thiện chất lượng của các vector từ nhưng cũng có nguy cơ dẫn đến hiện tượng overfitting nếu số lần lặp quá nhiều so với dữ liệu.

# sg (skip-gram): Một lựa chọn giữa hai kiến trúc mô hình: skip-gram (sg=1) và CBOW (continuous bag of words, sg=0). Skip-gram phù hợp với dữ liệu ít và có thể biểu diễn tốt các từ hiếm, trong khi CBOW nhanh hơn và hiệu quả hơn với dữ liệu lớn.

# Do ngữ liệu chỉ tập trung vào bộ luật lao động, nên sử dụng skip-gram cho phù hợp
# vector_size=100: Mỗi từ được biểu diễn bởi một vector 100 chiều.
# window=5: Mô hình sẽ xem xét 5 từ liền trước và 5 từ liền sau từ hiện tại khi xác định ngữ cảnh.
# min_count=1: Chỉ bao gồm các từ xuất hiện ít nhất một lần trong corpus.
# sg=1: Chọn kiến trúc Skip-gram cho mô hình.
# workers=4: Sử dụng 4 luồng để tăng tốc độ huấn luyện
model_word2vec = Word2Vec(vector_size=100, window=10, min_count=1, sg=1, workers=4)

#Xây dựng từ điển
# Phương thức này được sử dụng để xây dựng từ điển (hay còn gọi là bộ từ vựng) từ dữ liệu đầu vào. "Từ điển" ở đây là tập hợp các từ duy nhất được mô hình nhận diện và học trong quá trình huấn luyện.
model_word2vec.build_vocab(split_context_pre_train)

#Train model
# split_context_pre_train: Danh sách ngữ liệu.
# total_examples: Tổng số câu trong dữ liệu huấn luyện. Thông thường, tham số này được đặt tự động bởi model.corpus_count, vốn là số lượng câu được cung cấp khi xây dựng từ điển.
# epochs: Số lần lặp qua toàn bộ tập dữ liệu trong quá trình huấn luyện. Số lượng epochs càng cao, mô hình càng có nhiều cơ hội để học từ dữ liệu, nhưng cũng có nguy cơ dẫn đến overfitting nếu số lượng quá lớn.
model_word2vec.train(split_context_pre_train, total_examples=model_word2vec.corpus_count, epochs=100)

# Kết quả
# Biểu diễn vector của từ chương
vector = model_word2vec.wv['chương']

# Tìm ra 5 từ có ngữ cảnh gần nhất với từ "chương"
similarWord = model_word2vec.wv.most_similar('chương', topn=5)
print(similarWord)
# Lưu mô hình Word2Vec đã huấn luyện
# model_word2vec.save("word2vec.model")
