import os
from flask_cors import CORS
from flask import Flask
from flask import request
from flask import jsonify
from bert2vec_model import load_model, predict_result

FINE_TUNING_MODEL_PATH = "codes/static/fine_tuning_model.model"
BATCH_LIMIT = 10
# 在app启动之前加载模型和tokenizer
net, tokenizer = load_model(FINE_TUNING_MODEL_PATH)
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = './static/upload_files'

PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <form action="http://localhost:8080/upload_file" method="POST" enctype="multipart/form-data" >
        <input type="file" name="file"  multiple="multiple"/>
        <input type="submit" value="提交" />
    </form>
</body>
</html>'''


@app.route("/")
def getPage():
    return PAGE


@app.route("/upload_file", methods=['POST'])
def upload_file_and_predict():
    # 接收文件并存储到临时文件夹
    upload_file = request.files.getlist('fileUploads')
    for file in upload_file:
        if file and file.filename.endswith('.txt'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    doc_names = []
    docs = []
    # 将文件以字符的形式加载到内存中
    for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
        if file_name.endswith('.txt'):
            doc_names.append(file_name)
            doc_sentences = []
            with open(os.path.join(app.config['UPLOAD_FOLDER'], file_name), 'r', encoding='utf-8') as doc_file:
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
    start = 0
    result = []
    while len(sentences) - start > BATCH_LIMIT:
        result.extend(predict_result(net, tokenizer, sentences[start:start + BATCH_LIMIT]))
        start += BATCH_LIMIT
    result.extend(predict_result(net, tokenizer, sentences[start:len(sentences)]))
    # 重新转化为文章组织形式
    doc_labels = []
    index = 0
    for i in range(len(docs)):
        doc_label = []
        for j in range(len(docs[i])):
            doc_label.append(str(result[index]))
            index += 1
        doc_labels.append(doc_label)
    # 删除所有文件
    for file_name in doc_names:
        print("删除文件", os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file_name)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
    # 输出标签和内容
    # for i in range(len(docs)):
    #     print(doc_names[i]+":")
    #     for j in range(len(docs[i])):
    #         print("第", j, "段，标签为 ", doc_labels[i][j], ":", docs[i][j])
    result = {
        'file_name': doc_names,
        'file_labels': doc_labels,
        'file_sentence': docs
    }
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    # app.run(host, port, debug, options)
    app.run(debug=True)
