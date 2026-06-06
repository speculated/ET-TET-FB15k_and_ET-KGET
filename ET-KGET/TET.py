import torch
import torch.nn as nn
import trm

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm

def deterministic_index_select(input_tensor, dim, indices):
    """
    input_tensor: Tensor
    dim: dim
    indices: 1D tensor
    """
    tensor_transpose = torch.transpose(input_tensor, 0, dim)
    return tensor_transpose[indices].transpose(dim, 0)

class TET(nn.Module):
    def __init__(self, args, num_entities, num_rels, num_types, num_cluster):
        super(TET, self).__init__()
        self.embedding_dim = 300 # 100
        self.embedding_range = 0.1 # 10 / self.embedding_dim
        self.num_rels = num_rels + num_cluster # 1345 + 1081
        self.use_cuda = True
        self.dataset = 'YAGO43kET'
        # self.sample_ent2pair_size = 6 # 6
        self.tt_ablation = 'all' # all
        self.pooling = 'avg' # avg
        self.device = torch.device('cuda')
        self.num_nodes = num_entities + num_types # 14951 + 3584

        self.layer = TETLayer(args, self.embedding_dim, num_types, 0.5)

        self.entity = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim)) # (14951 + 3584, 100)
        nn.init.uniform_(tensor=self.entity, a=-self.embedding_range, b=self.embedding_range)
        self.relation = nn.Parameter(torch.randn(self.num_rels, self.embedding_dim)) # (1345 + 1081, 100) ... has reverse rels
        nn.init.uniform_(tensor=self.relation, a=-self.embedding_range, b=self.embedding_range)

        self.bert_nlayer = 3 # 3
        self.bert_nhead = 4 # 4
        self.bert_ff_dim = 480 # 480
        self.bert_activation = 'gelu' # gelu
        self.bert_hidden_dropout = 0.2 # 0.2
        self.bert_attn_dropout = 0.2 # 0.2
        self.local_pos_size = 200 # 200
        self.bert_layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
        self.local_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim)) # 1, 100
        torch.nn.init.normal_(self.local_cls, std=self.embedding_range)
        self.local_pos_embeds = nn.Embedding(self.local_pos_size, self.embedding_dim) # 200, 100
        torch.nn.init.normal_(self.local_pos_embeds.weight, std=self.embedding_range)
        bert_config = BertConfig(0, hidden_size=self.embedding_dim, # 100
                                 num_hidden_layers=self.bert_nlayer // 2, # 1
                                 num_attention_heads=self.bert_nhead, # 4
                                 intermediate_size=self.bert_ff_dim, # 480
                                 hidden_act=self.bert_activation, # gelu
                                 hidden_dropout_prob=self.bert_hidden_dropout, # 0.2
                                 attention_probs_dropout_prob=self.bert_attn_dropout, # 0.2
                                 max_position_embeddings=0,
                                 type_vocab_size=0,
                                 initializer_range=self.embedding_range) # 0.1
        self.bert_encoder = BertEncoder(bert_config)

        '''
        self.pair_layer = 3 # 3
        self.pair_head = 4 # 4
        self.pair_dropout = 0.2 # 0.2
        self.pair_ff_dim = 480 # 480
        self.pair_pos_embeds = nn.Embedding(1 + 2*self.sample_ent2pair_size, self.embedding_dim) # (1 + 2 * 6, 100)
        torch.nn.init.normal_(self.pair_pos_embeds.weight, std=self.embedding_range)
        # 100, 4, 480, 0.2
        pair_encoder_layers = TransformerEncoderLayer(self.embedding_dim, self.pair_head, self.pair_ff_dim, self.pair_dropout)
        # 3
        self.pair_encoder = TransformerEncoder(pair_encoder_layers, self.pair_layer)
        '''

    def convert_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def forward(self, et_content, kg_content, flag): # sample_ent2pair
        # shape(128, 3, 3)[:, :, 2] -> shape(128, 3)
        batch_size, et_neighbor_size = et_content[:, :, 2].size()
        # shape(128, 3, 100) deterministic_index_select(.) -> self.entity[[et_content[:, :, 2].view(-1)]]
        et_types = deterministic_index_select(self.entity, 0, et_content[:, :, 2].view(-1)).view(batch_size, et_neighbor_size, -1)
        et_relations_types = et_content[:, :, 1] # shape(128, 3)
        # shape(128, 3, 100)
        # deterministic_index_select(.) -> self.relation[[et_relations_types.view(-1) % self.num_rels]]
        et_relations = deterministic_index_select(self.relation, 0, et_relations_types.view(-1) % self.num_rels).view(batch_size, et_neighbor_size, -1)
        # 相反的关系，向量值均乘以-1
        et_relations[et_relations_types >= self.num_rels] = et_relations[et_relations_types >= self.num_rels] * -1

        '''
        if 'YAGO' in self.dataset:
            # for YAGO dataset, we should use cluster and type context pair to represent the KG relation
            batch_size, kg_neighbor_size = kg_content[:, :, 2].size()
            kg_entities = deterministic_index_select(self.entity, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, -1)
            _, pair_neighbor_size, _ = sample_ent2pair.size()
            kg_entity2pair = deterministic_index_select(sample_ent2pair, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, pair_neighbor_size, -1)
            pair_cluster = kg_entity2pair[:, :, :, 0]
            pair_type = kg_entity2pair[:, :, :, 1]
            pair_cluster_embs = deterministic_index_select(self.relation, 0, pair_cluster.view(-1)).view(-1, pair_neighbor_size, self.embedding_dim)
            pair_type_embs = deterministic_index_select(self.entity, 0, pair_type.view(-1)).view(-1, pair_neighbor_size, self.embedding_dim)
            kg_relations_types = kg_content[:, :, 1]
            kg_relations = deterministic_index_select(self.relation, 0, kg_relations_types.view(-1) % self.num_rels).view(-1, 1, self.embedding_dim)
            pairs = torch.cat((pair_cluster_embs, pair_type_embs), 2).view(-1, 2 * pair_cluster_embs.shape[1], pair_cluster_embs.shape[2])
            ent_pairs = torch.cat([kg_relations, pairs], 1).transpose(1, 0)  # [1 + num_pairs, bs, emb_dim]
            ent_pairs_pos = torch.arange(ent_pairs.shape[0], dtype=torch.long, device=self.device).repeat(ent_pairs.shape[1], 1)
            ent_pairs_pos_embeddings = self.pair_pos_embeds(ent_pairs_pos).transpose(1, 0)
            ent_pairs_embs = ent_pairs + ent_pairs_pos_embeddings
            mask = torch.zeros((ent_pairs_embs.shape[1], ent_pairs_embs.shape[0])).bool().to(self.device)
            x = self.pair_encoder(ent_pairs_embs, src_key_padding_mask=mask)

            if self.pooling == 'max':
                x, _ = torch.max(x, dim=0)
            elif self.pooling == "avg":
                x = torch.mean(x, dim=0)
            elif self.pooling == "min":
                x, _ = torch.min(x, dim=0)
            kg_relations = x.view(batch_size, -1, self.embedding_dim)
            kg_relations[kg_relations_types >= self.num_rels] = kg_relations[kg_relations_types >= self.num_rels] * -1
        else:
        '''

        batch_size, kg_neighbor_size = kg_content[:, :, 2].size()
        # shape(128, 7, 100) deterministic_index_select(.) -> self.entity[[et_content[:, :, 2].view(-1)]]
        kg_entities = deterministic_index_select(self.entity, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, -1)
        kg_relations_types = kg_content[:, :, 1]
        # shape(128, 7, 100)
        # deterministic_index_select(.) -> self.relation[[et_relations_types.view(-1) % self.num_rels]]
        kg_relations = deterministic_index_select(self.relation, 0, kg_relations_types.view(-1) % self.num_rels).view(batch_size, kg_neighbor_size, -1)
        # # 相反的关系，向量值均乘以-1
        kg_relations[kg_relations_types >= self.num_rels] = kg_relations[kg_relations_types >= self.num_rels] * -1

        # 3 et_types,  3 et_relations -> 2 et_types, 1 et_type 1 et_relation, 2 et_relations --hongbin
        # shape(128 * 3, 2, 100)
        # et_merge = torch.cat([et_types, et_relations], dim=1).view(-1, 2, self.embedding_dim)
        et_merge = torch.stack([et_relations, et_types], dim=2).view(-1, 2, self.embedding_dim)
        # print('et_content: ' + str(et_content[:, :, 2]))
        # print('et_content: ' + str(et_content[:, :, 2].size()))
        # print('et_merge: ' + str(et_merge))
        # print('et_merge: ' + str(et_merge.size()))
        # shape(128 * 3, 3, 100) unsqueeze(0) -> add one dimension: shape(3, 100) -> shape(1, 3, 100)
        et_pos = self.local_pos_embeds(torch.arange(0, 3, device=self.device)).unsqueeze(0).repeat(et_merge.shape[0], 1, 1)
        # shape(128 * 3, 3, 100)
        # shape(1, 100).expand(et_merge.size(0), 1, self.embedding_dim) -> shape(128 * 3, 1, 100)
        et_merge = torch.cat([self.local_cls.expand(et_merge.size(0), 1, self.embedding_dim), et_merge], dim=1) + et_pos
        et_merge = self.bert_layer_norm(et_merge)
        # shape(128, 3, 100)
        # new_ones(.) -> shape(128* 3, 3); self.convert_mask(shape(128 * 3, 3)) -> shape(128, 1, 1, 3)
        # self.bert_encoder(.)[-1] -> shape(128 * 3, 3, 100); self.bert_encoder(.)[-1][:, 0, :] -> shape(128 * 3, 100)
        et_merge = self.bert_encoder(et_merge, self.convert_mask(et_merge.new_ones(et_merge.size(0), et_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)

        # shape(128 * 7, 2, 100) --hongbin
        # kg_merge = torch.cat([kg_entities, kg_relations], dim=1).view(-1, 2, self.embedding_dim)
        kg_merge = torch.stack([kg_relations, kg_entities], dim=2).view(-1, 2, self.embedding_dim)
        kg_pos = self.local_pos_embeds(torch.arange(0, 3, device=self.device)).unsqueeze(0).repeat(kg_merge.shape[0], 1, 1)
        kg_merge = torch.cat([self.local_cls.expand(kg_merge.size(0), 1, self.embedding_dim), kg_merge], dim=1) + kg_pos
        kg_merge = self.bert_layer_norm(kg_merge)
        # shape(128, 7, 100)
        kg_merge = self.bert_encoder(kg_merge, self.convert_mask(kg_merge.new_ones(kg_merge.size(0), kg_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)

        if self.tt_ablation == 'all':
            # shape(128, 3 + 3 + 7 + 7, 100)
            # et_kg_merge = torch.cat([et_types, et_relations, kg_entities, kg_relations], dim=1).view(batch_size, -1, self.embedding_dim)
            et_kg_merge_et = torch.stack([et_relations, et_types], dim=2)
            et_kg_merge_kg = torch.stack([kg_relations, kg_entities], dim=2)
            et_kg_merge = torch.cat([et_kg_merge_et, et_kg_merge_kg], dim=1).view(batch_size, -1, self.embedding_dim)
        elif self.tt_ablation == 'triple':
            et_kg_merge = torch.cat([kg_entities, kg_relations], dim=1).view(batch_size, -1, self.embedding_dim)
        elif self.tt_ablation == 'type':
            et_kg_merge = torch.cat([et_types, et_relations], dim=1).view(batch_size, -1, self.embedding_dim)

        _, et_kg_size, _ = et_kg_merge.size() # 20
        if et_kg_size >= self.local_pos_size-1:
            et_kg_merge = et_kg_merge[:, 0:self.local_pos_size-1, :]
            et_kg_size = self.local_pos_size-1
        # shape(128, 20, 100)
        et_kg_pos = self.local_pos_embeds(torch.arange(0, et_kg_size + 1, device=self.device)).unsqueeze(0).repeat(et_kg_merge.shape[0], 1, 1)
        # shape(128, 21, 100)
        et_kg_merge = torch.cat([self.local_cls.expand(et_kg_merge.size(0), 1, self.embedding_dim), et_kg_merge], dim=1) + et_kg_pos
        et_kg_merge = self.bert_layer_norm(et_kg_merge)
        # shape(128, 1, 100)
        et_kg_merge = self.bert_encoder(et_kg_merge, self.convert_mask(et_kg_merge.new_ones(et_merge.size(0), et_kg_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1]
        if flag:
            et_kg_merge_type = torch.empty(0, device=self.device)
            et_kg_merge_entity = torch.empty(0, device=self.device)
            # et_kg_merge_entity_all = torch.empty(0, device=self.device)
            for i in range(1, et_neighbor_size * 2, 2):
                if et_kg_merge[:, i] is None:
                    break
                et_kg_merge_type = torch.cat([et_kg_merge_type, et_kg_merge[:, i].unsqueeze(1)], dim=1)
            for j in range(et_neighbor_size * 2 + 1, et_kg_size, 2):
                if et_kg_merge[:, j] is None:
                    break
                et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, j].unsqueeze(1)], dim=1)
            # for k in range(8, et_kg_size + 1, 2):
            #     if et_kg_merge[:, k] is None:
            #         break
            #     et_kg_merge_entity_all = torch.cat([et_kg_merge_entity_all, et_kg_merge[:, k].unsqueeze(1)], dim=1)
        else:
            et_kg_merge_type = None
            et_kg_merge_entity = None

        # if flag:
        #     et_kg_merge_type = torch.cat([et_kg_merge[:, 1], et_kg_merge[:, 3]], dim=1)
        #     et_kg_merge_type = torch.cat([et_kg_merge_type, et_kg_merge[:, 5]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge[:, 7], et_kg_merge[:, 9]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 11]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 13]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 15]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 17]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 19]], dim=1)
        #     et_kg_merge_entity = torch.cat([et_kg_merge_entity, et_kg_merge[:, 21]], dim=1)
        # else:
        #     et_kg_merge_type = None
        #     et_kg_merge_entity = None
        et_kg_merge = et_kg_merge[:, 0].unsqueeze(1)

        if self.tt_ablation == 'all':
            local_embedding = torch.cat([et_merge, kg_merge], dim=1) # (128, 3 + 7, 100)
        elif self.tt_ablation == 'triple':
            local_embedding = kg_merge
        elif self.tt_ablation == 'type':
            local_embedding = et_merge
        # global_embedding = et_kg_merge
        output, context_embedding, second_local_type, second_local_entity = self.layer(local_embedding, et_kg_merge, et_neighbor_size, flag)

        return output, context_embedding, second_local_type, et_kg_merge_type, second_local_entity, et_kg_merge_entity


class TETLayer(nn.Module):
    def __init__(self, args, embedding_dim, num_types, temperature):
        super(TETLayer, self).__init__()
        self.embedding_dim = embedding_dim # 100
        self.num_types = num_types # 3584
        self.fc = nn.Linear(embedding_dim, num_types)
        self.temperature = temperature # 0.5
        self.device = torch.device('cuda')
        self.dataset = 'YAGO43kET' # --hongbin

        self.trm_nlayer = 3 # 3
        self.trm_nhead = 4 # 4
        self.trm_hidden_dropout = 0.2 # 0.2
        self.trm_attn_dropout = 0.2 # 0.2
        self.trm_ff_dim = 480 # 480
        self.global_pos_size = 200 # 200
        self.embedding_range = 0.1 # 10 / self.embedding_dim

        self.global_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim)) # 1, 100
        torch.nn.init.normal_(self.global_cls, std=self.embedding_range)
        self.pos_embeds = nn.Embedding(self.global_pos_size, self.embedding_dim) # 200, 100
        torch.nn.init.normal_(self.pos_embeds.weight, std=self.embedding_range)
        self.layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
        self.loc_glo_weight = nn.Parameter(torch.ones(1))

        self.transformer_encoder = trm.Encoder(
            lambda: trm.EncoderLayer(
                self.embedding_dim, # 100
                trm.MultiHeadedAttentionWithRelations(
                    self.trm_nhead, # 4
                    self.embedding_dim, # 100
                    self.trm_attn_dropout), # 0.2
                trm.PositionwiseFeedForward(
                    self.embedding_dim, # 100
                    self.trm_ff_dim, # 480
                    self.trm_hidden_dropout), # 0.2
                num_relation_kinds=0,
                dropout=self.trm_hidden_dropout), # 0.2
            self.trm_nlayer, # 3
            self.embedding_range, # 0.1
            tie_layers=False)

    def convert_mask_trm(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def forward(self, local_embedding, et_kg_merge, et_neighbor_size, flag):
        local_msg = torch.relu(local_embedding)
        predict1 = self.fc(local_msg) # shape(128, 10, 3584)

        batch_size, neighbor_size, emb_size = local_embedding.size() # 128, 10, 100
        attention_mask = torch.ones(batch_size, neighbor_size + 1).bool().to(self.device) # shape(128, 11)
        # shape(128, 11, 100)
        second_local = torch.cat([self.global_cls.expand(batch_size, 1, emb_size), local_embedding], dim=1)
        # shape(3, 100)
        pos = self.pos_embeds(torch.arange(0, 2).to(self.device))
        # shape(128, 100) + shape(1, 100)
        second_local[:, 0] = second_local[:, 0] + pos[0].unsqueeze(0) # [:, 0] == [:, 0, :]
        # shape(128, 100) + shape(1, 100)
        # second_local[:, 1] = second_local[:, 1] + pos[1].unsqueeze(0)
        # shape(128, 9, 100) + shape(1, 1, 100)
        second_local[:, 1:] = second_local[:, 1:] + pos[1].view(1, 1, -1)
        # shape(128, 11, 100)
        second_local = self.layer_norm(second_local)
        # shape(128, 1, 100)
        # second_local[-1][:, :2] -> shape(128, 2, 100); second_local[-1][:, :2][:, 0] -> shape(128, 100)
        # abc = self.convert_mask_trm(attention_mask)
        # self.convert_mask_trm(shape(128, 11)) -> shape(128, 11, 11)

        second_local = self.transformer_encoder(second_local, None, self.convert_mask_trm(attention_mask))
        second_local = second_local[-1]
        if flag:
            second_local_type = torch.empty(0, device=self.device)
            second_local_entity = torch.empty(0, device=self.device)
            for i in range(1, et_neighbor_size + 1):
                if second_local[:, i] is None:
                    break
                second_local_type = torch.cat([second_local_type, second_local[:, i].unsqueeze(1)], dim=1)
            for j in range(et_neighbor_size + 1, neighbor_size + 1):
                if second_local[:, j] is None:
                    break
                second_local_entity = torch.cat([second_local_entity, second_local[:, j].unsqueeze(1)], dim=1)
            # second_local_type = torch.cat([second_local[:, 1].unsqueeze(1), second_local[:, 2].unsqueeze(1)], dim=1)
            # second_local_type = torch.cat([second_local_type, second_local[:, 3].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local[:, 4].unsqueeze(1), second_local[:, 5].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 6].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 7].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 8].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 9].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 10].unsqueeze(1)], dim=1)
            # second_local_entity = torch.cat([second_local_entity, second_local[:, 11].unsqueeze(1)], dim=1)
        else:
            second_local_type = None
            second_local_entity = None
        second_local = second_local[:, 0].unsqueeze(1)
        predict2 = self.fc(torch.relu(second_local)) # (128, 1, 3584)

        predict3 = self.fc(torch.relu(et_kg_merge))  # (128, 1, 3584)

        # if flag:
        #     attention_mask_two = torch.ones(batch_size, neighbor_size).bool().to(self.device)
        #     et_kg_merge_global = et_kg_merge_global + pos[1].view(1, 1, -1)
        #     et_kg_merge_global = self.layer_norm(et_kg_merge_global)
        #     second_global = self.transformer_encoder(et_kg_merge_global, None, self.convert_mask_trm(attention_mask_two))
        #     second_global = second_global[-1]
        #     second_global_type = torch.empty(0, device=self.device)
        #     second_global_entity = torch.empty(0, device=self.device)
        #     for i in range(0, 3):
        #         if second_global[:, i] is None:
        #             break
        #         second_global_type = torch.cat([second_global_type, second_global[:, i].unsqueeze(1)], dim=1)
        #     for j in range(3, neighbor_size):
        #         if second_global[:, j] is None:
        #             break
        #         second_global_entity = torch.cat([second_global_entity, second_global[:, j].unsqueeze(1)], dim=1)
        # else:
        #     second_global_type = None
        #     second_global_entity = None

        predict = torch.cat([predict1, predict2, predict3], dim=1) # (128, 10 + 1 + 1, 3584)
        weight = torch.softmax(self.temperature * predict, dim=1)
        # 使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变
        predict = (predict * weight.detach()).sum(1).sigmoid()

        return predict, second_local, second_local_type, second_local_entity