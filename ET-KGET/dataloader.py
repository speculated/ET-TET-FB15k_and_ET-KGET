import pickle
import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class ETDataset(Dataset):
    def __init__(self, args, data_name, e2id, r2id, t2id, c2id, data_flag):
        self.args = args
        self.data_name = data_name
        self.e2id = e2id
        self.r2id = r2id
        self.t2id = t2id
        self.c2id = c2id
        self.sample_et_size = args["sample_et_size"] # 3
        self.sample_kg_size = args["sample_kg_size"] # 7
        # [([[e2id, c2id+len(r2id), t2id+len(e2id)]], [[e2id, r2id / len(r2id)+len(c2id)+r2id, e2id]], e2id)]
        self.data = self.load_dataset()
        self.data_flag = data_flag
        # self.test_all_kg = args['test_all_kg']
        # self.test_all_et = args['test_all_et']

    def load_dataset(self):
        data_name_path = './data' + '/' + 'YAGO43kET' + '/' + self.data_name
        contents = []

        output_pickle = data_name_path[0: data_name_path.rfind('.')] + '.pkl'
        if os.path.exists(output_pickle):
            with open(output_pickle, 'rb') as handle:
                contents = pickle.load(handle)
                '''
                contents_add = []
                for i in contents:
                    # 增加类型的簇边
                    # add_clu = []
                    # for j in i[0]:
                    #     add_clu.append([j[0], j[1], j[2]])
                    #     for clu_id, clu_index in zip(range(len(cluster_index)), cluster_index):
                    #         if j[2] - len(self.e2id) in clu_index and clu_id + len(self.r2id) != j[1]:
                    #             add_clu.append([j[0], clu_id + len(self.r2id), j[2]])
                    # 出边、存在零出边(用入边的反向边代替)
                    del_rel_inv = []  # 116; 37; 36
                    tail_entities = []
                    for k in i[1]:  # delect relation_inv
                        if k[1] < len(self.r2id):
                            del_rel_inv.append([k[0], k[1], k[2]])
                            tail_entities.append(k[2])
                    if len(del_rel_inv) == 0:
                        for k_1 in i[1]:
                            del_rel_inv.append([k_1[0], k_1[1], k_1[2]])
                    else:
                        for k_2 in i[1]:
                            if k_2[2] not in tail_entities and k_2[1] > len(self.r2id):
                                del_rel_inv.append([k_2[0], k_2[1], k_2[2]])
                    contents_add.append((i[0], del_rel_inv, i[2]))
                '''
                return contents

        with open(data_name_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                mask_ent, et_triples, kg_triples, clu_triples = [_.strip() for _ in line.split('|||')]
                et_content_list = et_triples.split(' [SEP] ')
                et_list = []

                kg_content_list = kg_triples.split(' [SEP] ')
                kg_list = []

                for et_content in et_content_list:
                    et_head, et_rel, et_type = et_content.split(' ')
                    et_head_id = self.e2id[et_head]
                    et_type_id = self.t2id[et_type] + len(self.e2id)
                    # [[e2id, c2id+len(r2id), t2id+len(e2id)]]
                    et_list.append([et_head_id, self.c2id[et_rel] + len(self.r2id), et_type_id])

                for kg_content in kg_content_list:
                    kg_head, kg_rel, kg_tail = kg_content.split(' ')
                    if kg_rel.startswith('inv-'):
                        # inverse relation: len(r2id)+len(c2id)+r2id
                        kg_rel_id = len(self.r2id) + len(self.c2id) + self.r2id[kg_rel[4:]]
                    else:
                        kg_rel_id = self.r2id[kg_rel]
                    kg_head_id = self.e2id[kg_head]
                    kg_tail_id = self.e2id[kg_tail]
                    # [[e2id, r2id / len(r2id)+len(c2id)+r2id, e2id]]
                    kg_list.append([kg_head_id, kg_rel_id, kg_tail_id])

                # [([[e2id, c2id+len(r2id), t2id+len(e2id)]], [[e2id, r2id / len(r2id)+len(c2id)+r2id, e2id]], e2id)]
                contents.append((et_list, kg_list, self.e2id[mask_ent]))

        with open(output_pickle, 'wb') as handle:
            pickle.dump(contents, handle)

        return contents

    # 此功能是当使用带参数的函数调用时，会自动将参数对应到相应的参数
    def __getitem__(self, index):
        et_content = self.data[index][0]
        kg_content = self.data[index][1]
        ent = self.data[index][2]

        # sample et_neighbor
        single_et_np_list = []
        if self.sample_et_size != 1:
            sampled_index = np.random.choice(range(0, len(et_content)), size=self.sample_et_size,
                                             replace=len(range(0, len(et_content))) < self.sample_et_size)
            for i in sampled_index:
                single_et_np_list.append(et_content[i])
        else:
            single_et_np_list.append(et_content[0])

        # sample kg_neighbor
        single_kg_np_list = []
        if self.sample_kg_size != 1:
            sampled_index = np.random.choice(range(0, len(kg_content)), size=self.sample_kg_size,
                                             replace=len(range(0, len(kg_content))) < self.sample_kg_size)
            for i in sampled_index:
                single_kg_np_list.append(kg_content[i])
        else:
            single_kg_np_list.append(kg_content[0])

        all_et = et_content
        all_kg = kg_content
        sample_et = single_et_np_list
        sample_kg = single_kg_np_list

        gt_ent = ent

        # if self.data_flag == 'test':
        #     # for test, we need all neighbor information
        #     # Nevertheless, using all neighbor information directly needs a considerable GPU memory which is not
        #     # supported by mainstream GPUs. Here we limit the max. number of kg neighbors to 200 and max. num of et
        #     # neighbors to 100.
        #     if len(all_kg) > self.test_all_kg:
        #         sampled_kg_index = np.random.choice(range(0, len(kg_content)), size=self.test_all_kg, replace=False)
        #         all_kg = []
        #         for i in sampled_kg_index:
        #             all_kg.append(kg_content[i])
        #     if len(all_et) > self.test_all_et:
        #         sampled_et_index = np.random.choice(range(0, len(et_content)), size=self.test_all_et, replace=False)
        #         all_et = []
        #         for i in sampled_et_index:
        #             all_et.append(et_content[i])
        #     return all_et, all_kg, gt_ent
        # else:
        #     return sample_et, sample_kg, gt_ent

        if self.data_flag == 'test':
            # for test, we need all neighbor information
            return all_et, all_kg, gt_ent
        else:
            # [[]]
            return sample_et, sample_kg, gt_ent

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sample_et_content_list = []
        sample_et_content_list.append([_[0] for _ in batch])

        sample_kg_content_list = []
        sample_kg_content_list.append([_[1] for _ in batch])

        gt_ent_list = []
        gt_ent_list.append([_[2] for _ in batch])

        et_content = torch.LongTensor(sample_et_content_list[0])
        kg_content = torch.LongTensor(sample_kg_content_list[0])

        gt_ent = torch.LongTensor(gt_ent_list[0])

        # shape(1, len([[e2id, c2id+len(r2id), t2id+len(e2id)]]), 3),
        # shape(1, len([[e2id, r2id / len(r2id)+len(c2id)+r2id, e2id]]), 3)
        # shape(1)
        return et_content, kg_content, gt_ent
