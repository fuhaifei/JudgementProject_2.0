import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

# 哈工大中文bert模型,最高支持512长度句子
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
FINE_TUNING_MODEL_PATH = "codes/static/fine_tuning_model.model"
MAX_SEQ_LENGTH = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_train_data(sentences, tokenizer, max_seq_length=MAX_SEQ_LENGTH) -> object:
    """
    将输入句子转化为模型输入形式
    :param sentences: 输入段落
    :param tokenizer: 分词器
    :param max_seq_length: 最大输入序列长
    :return: 返回 序列， 序列mask， 序列type_ids
    """
    result_tokens = []
    result_masks = []
    result_token_ids = []
    for sentence in sentences:
        sentence = ''.join(sentence)
        # 对源句子进行切词和截断
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(sentence_tokens) > max_seq_length - 2:
            sentence_tokens = sentence_tokens[:max_seq_length - 2]
        sentence_tokens = ['[CLS]'] + sentence_tokens + ['[SEP]']
        sentence_tokens = tokenizer.convert_tokens_to_ids(sentence_tokens)
        # 获取对应的mask和segment()
        sentence_padding = [0] * (max_seq_length - len(sentence_tokens))
        sentence_mask = [1] * len(sentence_tokens) + sentence_padding
        sentence_type_ids = [0] * len(sentence_tokens) + sentence_padding
        sentence_tokens += sentence_padding
        result_tokens.append(sentence_tokens)
        result_masks.append(sentence_mask)
        result_token_ids.append(sentence_type_ids)
    return result_tokens, result_masks, result_token_ids


def load_model(model_path=FINE_TUNING_MODEL_PATH):
    net = torch.load(model_path, map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return net, tokenizer


def predict_result(net, tokenizer, inputs, device=DEVICE):
    """
    预测结果
    :param tokenizer: 分词器
    :param net: fine tuning结果
    :param inputs: 段落一维数组
    :param device: 设备
    :return: 预测结果一维数组
    """
    if device is None:
        device = list(net.parameters())[0].device

    input_tokens, input_masks, input_token_ids = prepare_train_data(inputs, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    input_tokens_tensor = torch.LongTensor(input_tokens).to(DEVICE)
    input_masks_tensor = torch.LongTensor(input_masks).to(DEVICE)
    input_type_ids_tensor = torch.LongTensor(input_token_ids).to(DEVICE)

    net.to(device)
    net.eval()
    # 预测结果
    prediction = net(input_tokens_tensor, input_masks_tensor, input_type_ids_tensor).logits
    result = prediction.argmax(dim=1).cpu().detach().numpy()
    torch.cuda.empty_cache()
    return result

# # 初始化数据
# docs, labels = load_label_data()
# doc_flatten = [sentence for doc in docs for sentence in doc]
# doc_labels_flatten = [sentence for doc in labels for sentence in doc]
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# result_tokens, result_masks, result_token_ids = prepare_train_data(doc_flatten, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
# train_iter, test_iter = get_iter_dataset(result_tokens, doc_labels_flatten, result_masks, result_token_ids, batch_size=BATCH_SIZE)
#
# # 初始化模型
# net = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=11)
# loss_func = nn.CrossEntropyLoss()
# param_optimizer = list(net.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# # 避免对bias 和 layerNorm层正则化
# optimizer_grouped_parameters = [
#     {
#         'params':
#         [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#         'weight_decay':
#         0.01
#     },
#     {
#         'params':
#         [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#         'weight_decay':
#         0.0
#     }
# ]
# optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)
#
# fine_tuning_model(net, loss_func, optimizer, train_iter, test_iter, num_epochs=EPOCHS)
# eval_net(net, test_iter, device=DEVICE)
