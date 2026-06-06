import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def relative_attention_logits(query, key, relation):
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    if relation == None:
        return qk_matmul / math.sqrt(query.shape[-1])

    q_t = query.permute(0, 2, 1, 3)
    r_t = relation.transpose(-2, -1)
    q_tr_t_matmul = torch.matmul(q_t, r_t)
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

def relative_attention_values(weight, value, relation):
    wv_matmul = torch.matmul(weight, value)

    if relation == None:
        return wv_matmul

    w_t = weight.permute(0, 2, 1, 3)
    w_tr_matmul = torch.matmul(w_t, relation)
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t

def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])

# def clones(module_fn, N):
#     layers = [module_fn() for _ in range(N)]
#     layers[1] = layers[-2] = module_fn()
#     layers[0] = layers[-1] = module_fn()
#     return nn.ModuleList(layers)

def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig

class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h # 100 // 4
        self.h = h # 4
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4) # (100, 100), 4
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) # 0.2

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # add one dimension: shape(x) -> shape(x, 1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        # 调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Encoder(nn.Module):
    def __init__(self, layer, N, initializer_range, tie_layers=False):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N) # lambda: trm.EncoderLayer(.), 3
        self.initializer_range = initializer_range # 0.1
        self.apply(self.init_bert_weights)

    def forward(self, x, relation, mask):
        all_x = []
        for layer in self.layers:
            x = layer(x, relation, mask)
            all_x.append(x)
        return all_x

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range) # 0.0, 0.1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(self.dropout(sublayer(x)) + x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # trm.MultiHeadedAttentionWithRelations(.)
        self.feed_forward = feed_forward # trm.PositionwiseFeedForward(.)
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2) # (100, 0.2), 2
        self.size = size # 100

        if num_relation_kinds != 0:
            self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
            self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        else:
            self.relation_k_emb = lambda x: None
            self.relation_v_emb = lambda x: None

    def forward(self, x, relation, mask):
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask)) # residual connection 1
        return self.sublayer[1](x, self.feed_forward) # residual connection 2

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 100, 480
        self.w_2 = nn.Linear(d_ff, d_model) # 480, 100
        self.dropout = nn.Dropout(dropout) # 0.2

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
