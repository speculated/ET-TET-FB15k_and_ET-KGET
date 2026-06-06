import torch
import logging
import os
import numpy as np

def set_logger(args):
    # 'save/SFNA'
    if not os.path.exists(os.path.join('save', 'SFNA')):
        os.makedirs(os.path.join(os.getcwd(), 'save', 'SFNA'))
    # 'save/SFNA/log'
    log_file = os.path.join('save', 'SFNA', 'log' +'.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def read_id(path):
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            tmp[e] = int(t)
    return tmp

def load_type_labels(paths, e2id, t2id):
    labels = torch.zeros(len(e2id), len(t2id))
    for path in paths:
        with open(path, encoding='utf-8') as r:
            for line in r:
                e, t = line.strip().split('\t')
                e_id, t_id = e2id[e], t2id[t]
                labels[e_id, t_id] = 1
    return labels

def load_id(path, e2id):
    ret = set()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            ret.add(e2id[e])
    return list(ret)

def load_train_all_labels(data_dir, e2id, t2id):
    train_type_label = load_type_labels([
        os.path.join(data_dir, 'ET_train.txt'),
        os.path.join(data_dir, 'ET_valid.txt')
    ], e2id, t2id)
    test_type_label = load_type_labels([
        os.path.join(data_dir, 'ET_train.txt'),
        os.path.join(data_dir, 'ET_valid.txt'),
        os.path.join(data_dir, 'ET_test.txt'),
    ], e2id, t2id).half() # float16

    return train_type_label, test_type_label

def load_entity_cluster_type_pair_context(args, r2id, e2id):
    data_name_path = './data' + '/' + 'YAGO43kET' + '/ent2pair.npy'
    sample_ent2pair_size = 6 # 6
    ent2pair = np.load(data_name_path, allow_pickle=True).tolist()
    sample_ent2pair = []
    for single_sample_ent2pair in ent2pair: # 14951: 816 / 680 / 811 / 1165 / 1029 / 1053 / 1085
        if single_sample_ent2pair == list([[0, 0]]):
            single_sample_ent2pair = list([[1081, 3584]])
        single_sample_ent2pair_list = []
        if sample_ent2pair_size != 1:
            sampled_index = np.random.choice(range(0, len(single_sample_ent2pair)), size=sample_ent2pair_size,
                                             replace=len(range(0, len(single_sample_ent2pair))) < sample_ent2pair_size)
            for i in sampled_index:
                clu_info = single_sample_ent2pair[i][0] + len(r2id)
                type_info = single_sample_ent2pair[i][1] + len(e2id)
                single_sample_ent2pair_list.append([clu_info, type_info])
        else:
            clu_info = single_sample_ent2pair[0][0] + len(r2id)
            type_info = single_sample_ent2pair[0][1] + len(e2id)
            single_sample_ent2pair_list.append([clu_info, type_info])
        sample_ent2pair.append(single_sample_ent2pair_list)
    return sample_ent2pair

def evaluate(path, predict, all_true, e2id, t2id):
    logs = []
    f = open('rank.txt', 'w', encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t] # e2id, t2id
            tmp = predict[e] - all_true[e]
            tmp[t] = predict[e, t]
            # torch.argsort: 返回排序后的值所对应原a的下标，即torch.sort()返回的indices
            argsort = torch.argsort(tmp, descending=True)
            ranking = (argsort == t).nonzero()
            assert ranking.size(0) == 1
            ranking = ranking.item() + 1
            print(line.strip(), ranking, file=f) # print into file
            logs.append({
                'MRR': 1.0 / ranking, # 作为底数，以降序排序，越靠前则值越大
                'MR': float(ranking), # 排序浮点数表示
                'HIT@1': 1.0 if ranking <= 1 else 0.0, # 严格要求排序第一，即答案概率最大
                'HIT@3': 1.0 if ranking <= 3 else 0.0, # 要求排序前三
                'HIT@10': 1.0 if ranking <= 10 else 0.0 # 要求排序前十
            })
    MRR = 0
    for metric in logs[0]:
        tmp = sum([_[metric] for _ in logs]) / len(logs) # 取平均
        if metric == 'MRR':
            MRR = tmp
        logging.debug('%s: %f' % (metric, tmp))
    return MRR

def slight_fna_loss(predict, label, beta):
    loss = torch.nn.BCELoss(reduction='none')
    output = loss(predict, label)
    positive_loss = output * label
    negative_weight = predict.detach().clone()
    small_ids = negative_weight <= 0.5
    large_ids = negative_weight > 0.5

    negative_weight[small_ids] = beta * (3 * negative_weight[small_ids] - 2 * negative_weight[small_ids].pow(2))
    negative_weight[large_ids] = beta * (negative_weight[large_ids] - 2 * negative_weight[large_ids].pow(2) + 1)

    # (1 - label) represent lable not in KG
    negative_weight = negative_weight * (1 - label)
    negative_loss = negative_weight * output
    return positive_loss.mean(), negative_loss.mean()
