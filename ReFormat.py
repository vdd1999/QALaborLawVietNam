import json

import uuid

from docx import Document

# Generate a UUID
# uid = uuid.uuid4()
# print(uid)

file_names = [ 
    'title_1',
    'title_2',
    'title_3',
    'title_4'
]

titles= [ 
    "Trường Đại học Công Thương Thành phố Hồ Chí Minh, từ ngày 01/07/2023",
    "Tuyển sinh, mã trường, tên ngành",
    "Quy chế đào tạo tín chỉ",
    "Các câu hỏi thường gặp"
]

title_obj = dict()
title_obj['title'] = titles[3]
title_obj['paragraphs'] = []

doc = Document(f'{file_names[3]}.docx')

crr_ctx = None

for para in doc.paragraphs:
    tmp = para.text.lower()
    print(tmp)
    print('-'*10)
    if 'context' in tmp:
        print('context')
        if crr_ctx:
            title_obj['paragraphs'].append(crr_ctx)
        crr_ctx = dict()
        crr_ctx['context'] = para.text.split(':')[1].strip()
        crr_ctx['qas'] = []
    elif 'question' in tmp:
        print('question')
        crr_qa = dict()
        crr_qa['question'] = para.text.split(':')[1].strip()
        crr_qa['id'] = str(uuid.uuid4())
        crr_qa['is_impossible'] = False
        crr_qa['answers'] = []
    elif 'answers' in tmp:
        print('answers')
        crr_ans = dict()
        crr_ans['text'] = para.text.split(':')[1].strip()
        crr_ans['answer_start'] = None
        crr_qa['answers'].append(crr_ans)
        crr_ctx['qas'].append(crr_qa)

if crr_ctx:
    title_obj['paragraphs'].append(crr_ctx)

# print(len(title_obj['paragraphs']))
# write json file
with open(f'{file_names[3]}.json', 'w+', encoding='utf-8') as outfile:
    json.dump(title_obj, outfile, ensure_ascii=False, indent=4)
