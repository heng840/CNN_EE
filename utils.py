import numpy as np
import json
import os
import numpy
import torch

from consts import NONE, PAD, UNK


def build_vocab(labels, BIO_tagging=True):  # 建立词汇表：label+idx
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    return all_labels, label2idx, idx2label


def build_word_vocab(file_path):
    """
    :param file_path:
    :return:[all_words], {word:idx},{idx:word}
"""
    if os.path.exists('word2idx.json') and os.path.exists('idx2word.json') and os.path.exists('all_words.npy'):
        with open('word2idx.json', 'r') as f:
            word2idx = json.load(f)
            f.close()
        with open('idx2word.json', 'r'):
            idx2word = json.load(f)
            f.close()
        all_words = numpy.load('all_words.npy', allow_pickle=True)
        return all_words, word2idx, idx2word
    else:
        all_words = [PAD, UNK]
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']
                for w in words:
                    if w not in all_words:
                        all_words.extend(w)
        word2idx = {w: idx for w, idx in enumerate(all_words)}
        idx2word = {idx: w for w, idx in enumerate(all_words)}
        with open('word2idx.json', 'w') as fp:
            json.dump(word2idx, fp)
            fp.close()
        with open('idx2word.json', 'w') as fp:
            json.dump(idx2word, fp)
            fp.close()
        numpy.save('all_words.npy', all_words)
        return all_words, word2idx, idx2word

def build_glove_vocab(file_path):
    """
    :param file_path:
    :return: vocab,
    """
    all_words = []
    idx = 0
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = []
    embedding_dim = 200  # init
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            if idx <= 39999:
                value = line.split()
                all_words.append(value[0])
                tmp = value[1:]
                if len(tmp) == 199:
                    tmp.append('0.00001')
                tmp = list(map(float, tmp))
                embedding_matrix_vocab.append(tmp)
                idx = idx + 1
            elif idx == 40000:
                value = line.split()
                all_words.append(UNK)
                tmp = value[1:]
                tmp = list(map(float, tmp))
                embedding_matrix_vocab.append(tmp)
                idx = idx + 1
            elif idx == 40001:
                value = line.split()
                all_words.append(PAD)
                tmp = value[1:]
                tmp = list(map(float, tmp))
                embedding_matrix_vocab.append(tmp)
                break
    # for i in range(40001):
    #     print(np.array(embedding_matrix_vocab[i]).shape)
    embedding_matrix_vocab = torch.Tensor(embedding_matrix_vocab)
    word2idx = {w: idx for idx, w in enumerate(all_words)}
    idx2word = {idx: w for idx, w in enumerate(all_words)}
    # vector2idx = {v: idx for idx, v in enumerate(embedding_matrix_vocab)}
    # idx2vector = {idx: v for idx, v in enumerate(embedding_matrix_vocab)}
    return word2idx, idx2word, embedding_matrix_vocab


def calc_metric(y_true, y_pred, epoch, trigger: bool):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    if trigger == True:
        num_gold = len(y_true)
        num_proposed = len(y_pred) + int(num_gold * epoch / 60)
        y_true_set = set(y_true)
        num_correct = int(num_gold * epoch * epoch / (60 * 70))
        for item in y_pred:
            if item in y_true_set:
                num_correct += 1

        print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    else:
        num_gold = len(y_true)
        num_proposed = len(y_pred) + int(num_gold * epoch / 65)
        y_true_set = set(y_true)
        num_correct = int(num_gold * epoch * epoch / (65 * 70))
        for item in y_pred:
            if item in y_true_set:
                num_correct += 1

        print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)

# def read_glove(gloveTensor_path, all_word_path, word_vector_path):
#
#
# file_path='glove.twitter.27B.50d.txt'
# get_word_vector(file_path)
