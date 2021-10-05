from codes.api.bert2vec_model import predict_result


def clf_sentences(sentences, batch_limit, net, tokenizer):
    """
    对进入的句子进行分类
    :param sentences: 传入的段落信息
    :param batch_limit: 每次预测最多的段落数量
    :param net: 分类器
    :param tokenizer: 分词器
    :return: 返回结果
    """
    # 每次预测batch_limit数量的句子
    start = 0
    result = []
    while len(sentences) - start > batch_limit:
        result.extend(predict_result(net, tokenizer,
                                     sentences[start:start + batch_limit]))
        start += batch_limit
    result.extend(predict_result(net, tokenizer, sentences[start:len(sentences)]))
    return result


def doc_checkout(doc_label, label_dic):
    """
    寻找不能存在的类型和非判决书自然段
    :param doc_label: 段落对应大的标签
    :param label_dic: 标签dic
    :return: 返回缺失的类型和非判决书类型(全部为字符穿类型，)
    """
    loss_types = []
    not_types = []
    labels = list(label_dic.keys())
    # 首先寻找缺少的类型（不包括最后一个非判决上诉自然段）
    print(doc_label)
    for i in range(0, len(labels) - 1):
        if labels[i] not in doc_label:
            print("i", labels[i])
            loss_types.append(labels[i])
    print(loss_types)
    # 寻找非判决书自然段
    for i in range(0, len(doc_label) - 1):
        if doc_label[i] + 1 == labels[len(labels) - 1]:
            not_types.append(doc_label[i])
    return loss_types, not_types

