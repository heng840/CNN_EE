import math

import torch
import torch.nn as nn
from CNN import CNN
from data_load import idx2trigger, argument2idx, word2idx, idx2word, embedding_matrix_vocab
from consts import NONE
from utils import find_triggers


class Net(nn.Module):  # Net类
    """
    input=tensor [batch_size,len(seq),len(vector)]
    """
    def __init__(self, device='cpu', trigger_size=None, argument_size=None):
        super(Net, self).__init__()
        self.device = device
        self.cnn = CNN(device=self.device, seq_len=128)
        hidden_size = 50+49+48+47
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size)
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size)
        )
        # 因为后续argument拼接了entity
        # self.fc = nn.Linear(self.in_features_fc(), 1)

    def predict_triggers(self, tokens_x_2d, triggers_y_2d, arguments_2d):
        """
        :param tokens_x_2d:
        :param triggers_y_2d:
        :param arguments_2d:
        :return:
        """
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        batch_size = tokens_x_2d.shape[0]

        "CNN"
        cnn_layer = self.cnn(tokens_x_2d)# torch.Size([24, outsize, 194])
        # bert:(batch_size, sequence_length, hidden_size),cnn output与之相同
        trigger_predict = self.fc_trigger(cnn_layer)
        trigger_predict_idx = trigger_predict.argmax(-1)

        "以下预测了argument，可以从predict_triggers中独立，用到了data_load里的idx2trigger"
        argument_hidden, argument_keys = [], []
        # print(cnn_layer)
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = cnn_layer[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_predict_idx[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = cnn_layer[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))  # 给定维度上对输入的张量序列seq 进行连接操作
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_predict, triggers_y_2d, trigger_predict_idx, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):  # 预测输入
        argument_hidden = torch.stack(argument_hidden)  # stack：沿着一个新维度对输入张量序列进行连接。序列中所有的张量都应该为相同形状，即是扩维拼接
        argument_logits = self.fc_argument(argument_hidden)  # linear
        argument_hat_1d = argument_logits.argmax(-1)  # 返回最大值

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:  # 与json文件可以对应
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,
                                                                                 argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d
