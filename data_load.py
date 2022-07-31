import numpy as np
from torch.utils import data
import json
from consts import NONE, PAD, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS
from utils import build_vocab, build_glove_vocab

# init vocab初始化词汇表
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
# all_words, word2idx, idx2word = build_word_vocab('data/train.json')
word2idx, idx2word, embedding_matrix_vocab = build_glove_vocab('glove.twitter.27B/glove.twitter.27B.200d.txt')


class ACE2005Dataset(data.Dataset):  # 数据类
    def __init__(self, fpath):
        self.sent_li, self.triggers_li, self.arguments_li = [], [], []  # sent：句子，sentence；arguments：元素
        with open(fpath, 'r') as f:
            data = json.load(f)  # dict:"sentence"..."word"
            for item in data:  # json数据包括了：words，拆分sentence
                words = item['words']
                triggers = [NONE] * len(words)
                arguments = {  # argument，创建空词典
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                for entity_mention in item['golden-entity-mentions']:  # 在data里，建立arguments，在candidates中添加三个
                    arguments['candidates'].append((entity_mention['start'], entity_mention['end'],
                                                    entity_mention['entity-type']))  # data里的词典entity_mention

                for event_mention in item['golden-event-mentions']:  # data里的词典
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):  # trigger
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (
                        event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:  # arguments词典
                        role = argument['role']
                        if role.startswith('Time'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头
                            role = role.split('-')[0]
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                self.sent_li.append(words)
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)

    def __len__(self):  # 返回长度
        return len(self.sent_li)

    def __getitem__(self, idx):  # 把类中的属性定义为序列，可以使用__getitem__()函数输出序列属性中的某个元素，这个方法返回与指定键关联的值
        words, triggers, arguments = self.sent_li[idx], self.triggers_li[idx], self.arguments_li[idx]
        # 这对应json数据里的词典
        # We give credits only to the first piece.
        tokens_x, is_heads = [], []  # to predict ->(seqlen * vector)
        for w in words:
            if w in word2idx:
                w_id = word2idx[w]
                tokens_x.append(w_id)
            else:
                tokens_x.append(word2idx[UNK])
        triggers_y = [trigger2idx[t] for t in triggers]  # true
        seqlen = len(tokens_x)
        return tokens_x, triggers_y, arguments, seqlen, words, triggers

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    """
    :param max_len:
    :param batch: batch_size
    :return: tensor(batch_size,*)
    """
    tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, words_2d, triggers_2d = list(map(list, zip(*batch)))  # 矩阵转置
    # maxlen = np.array(seqlens_1d).max()
    max_len = 128
    for i in range(len(tokens_x_2d)):
        """[0]*以后为什么还有数值：[0]*得到的是一个张量，用来补齐batch"""
        tokens_x_2d[i] = tokens_x_2d[i] + [word2idx[PAD]] * (max_len - len(tokens_x_2d[i]))  # 补零语句，batch
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (max_len - len(triggers_y_2d[i]))

    return tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, words_2d, triggers_2d
