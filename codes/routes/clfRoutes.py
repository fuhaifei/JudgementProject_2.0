import os
from flask import current_app, Blueprint
from flask import request
from flask import jsonify
from codes.api.bert2vec_model import predict_result
from codes.api.clfApi import clf_sentences, doc_checkout

clfApp = Blueprint('clf_blueprint', __name__)

PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <form action="http://localhost:8080/clf/upload_file" method="POST" enctype="multipart/form-data" >
        <input type="file" name="file"  multiple="multiple"/>
        <input type="submit" value="提交" />
    </form>
</body>
</html>'''


@clfApp.route("/")
def getPage():
    return PAGE


@clfApp.route("/upload_file", methods=['POST'])
def upload_file_and_predict():
    # 接收文件并存储到临时文件夹
    upload_file = request.files.getlist('fileUploads')
    for file in upload_file:
        if file and file.filename.endswith('.txt'):
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename))

    doc_names = []
    docs = []
    # 将文件以字符的形式加载到内存中
    for file_name in os.listdir(current_app.config['UPLOAD_FOLDER']):
        if file_name.endswith('.txt'):
            doc_names.append(file_name)
            doc_sentences = []
            with open(os.path.join(current_app.config['UPLOAD_FOLDER'], file_name), 'r', encoding='utf-8') as doc_file:
                sentence = doc_file.readline()
                while sentence is not None and sentence != '':
                    sentence = sentence.replace('\n', '').replace('\r', '').replace(' ', '')
                    if len(sentence) == 0:
                        sentence = doc_file.readline()
                        continue
                    doc_sentences.append(sentence)
                    sentence = doc_file.readline()
                docs.append(doc_sentences)

    sentences = [sentence for doc in docs for sentence in doc]
    # 每次预测batch_limit数量的句子
    result = clf_sentences(sentences, current_app.config['BATCH_LIMIT'], current_app.net, current_app.tokenizer)
    # 重新转化为文章组织形式
    doc_labels = []
    loss_types = []
    not_sentences = []
    index = 0
    for i in range(len(docs)):
        doc_label = []
        for j in range(len(docs[i])):
            doc_label.append(result[index])
            index += 1
        print(doc_label)
        loss_type, not_sentence = doc_checkout(doc_label, current_app.config['LABELS'])
        loss_types.append(list(map(str, loss_type)))
        not_sentences.append(list(map(str, not_sentence)))
        doc_labels.append(list(map(str, doc_label)))
    # 删除所有文件
    for file_name in doc_names:
        print("删除文件", os.path.join(current_app.config['UPLOAD_FOLDER'], file_name))
        if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)):
            os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'], file_name))
    result = {
        'file_name': doc_names,
        'file_labels': doc_labels,
        'file_sentence': docs,
        'file_loss_types': loss_types,
        'file_not_sentences': not_sentences
    }
    print(result)
    return jsonify(result)
