import torch
import torch.nn.functional as F
import torch.nn as nn


def ce_scl_loss(preds, ground_truth, hidden_state,
                lambda_value=0.5, temperature=1,
                weight=None, device='cuda'):
    if weight is not None:
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    ce_loss = cross_entropy(preds, ground_truth)
    c_loss = scl_loss(hidden_state, ground_truth, temperature)
    loss = torch.tensor(1 - lambda_value, device=device) * ce_loss + torch.tensor(lambda_value, device=device) * c_loss
    # print(ce_loss)
    # print(c_loss)
    # loss = (1-lambda_value) * ce_loss + lambda_value * c_loss
    return loss

def ce_scl_tcl_loss(preds, ground_truth, sequence_hidden_state, pool_hidden_state,
                lambda_value=0.5, temperature=1,
                weight=None, device='cuda'):
    if weight is not None:
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    ce_loss = cross_entropy(preds, ground_truth)
    c_loss = scl_loss(sequence_hidden_state, ground_truth, temperature)
    tc_loss = translation_cl_loss(pool_hidden_state, temperature)
    # print(ce_loss)
    # print(c_loss)
    # print(tc_loss)
    loss = torch.tensor(1 - lambda_value, device=device) * ce_loss + torch.tensor(lambda_value/2, device=device) * (c_loss + tc_loss)
    return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # revise to translate contrastive loss, where the current sentence itself and its translation is 1.
            mask = torch.eye(int(batch_size / 2), dtype=torch.float32)
            mask = torch.cat([mask, mask], dim=1)
            mask = torch.cat([mask, mask], dim=0)
            mask = mask.to(device)

        elif labels is not None:
            # 如果想要断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝。
            # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # fill 2D with 1 and 0 that is True and Flase respectively
            mask = torch.eq(labels, labels.T).float().to(device)

            # if cluster_lable is not None:
            #     cluster_lable = cluster_lable.contiguous().view(-1, 1)
            #     if cluster_lable.shape[0] != batch_size:
            #         raise ValueError('Num of labels does not match num of features')
            #     mask_cluster = torch.eq(cluster_lable, cluster_lable.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # 1
        # torch.unbind移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片, return tuple
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # shape(6, 2)
            anchor_count = contrast_count # 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits, every vector is dot product with others
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), # shape(6, 6)
            self.temperature)
        # for numerical stability, results from dot product are reduce with the maximum
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # shape(6, 1)
        logits = anchor_dot_contrast - logits_max.detach() # shape(6, 6)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # (1, 1)
        # mask-out self-contrast cases, torch.scatter表示在dim为1中，根据index的0填充到input中
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # 将对角线设置为0
        mask = mask * logits_mask

        # if cluster_lable is not None:
        #     # 将对角线设置为0
        #     mask_cluster = mask_cluster * logits_mask
        #     mask_cluster_mask = ~torch.eq(mask_cluster, mask)
        #     mask = mask_cluster_mask + mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20 # (6, 6)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


global_sup_con_loss = None


def get_sup_con_loss(device='cuda'):
    global global_sup_con_loss
    if global_sup_con_loss is None:
        global_sup_con_loss = SupConLoss(contrast_mode='all')
        global_sup_con_loss = nn.DataParallel(global_sup_con_loss)
        global_sup_con_loss.to(device)
    return global_sup_con_loss


def scl_loss(hidden_states, ground_truth, temperature=0.9):
    sup_con_loss = get_sup_con_loss()

    flatten_hidden_states = torch.reshape(hidden_states, (-1, 1, hidden_states.shape[-1]))
    if len(ground_truth.shape) == 1:
        flatten_ground_truth = ground_truth
    else:
        flatten_ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1])
    sup_con_loss.temperature = temperature

    loss = SupConLoss(contrast_mode='all', temperature=temperature).to('cuda')
    loss = loss(flatten_hidden_states, flatten_ground_truth)
    loss = loss / len(hidden_states)

    return loss

def translation_cl_loss(hidden_states, labels, temperature=0.9):
    sup_con_loss = get_sup_con_loss()
    # shape(6, 1, 2)
    flatten_hidden_states = torch.reshape(hidden_states, (-1, 1, hidden_states.shape[-1]))

    sup_con_loss.temperature = temperature

    loss = SupConLoss(contrast_mode='all', temperature=temperature).to('cuda')
    loss = loss(flatten_hidden_states, labels)
    loss = loss / len(hidden_states)

    return loss

if __name__ == '__main__':
    import numpy as np

    '''
    for t in np.arange(0, 1, 0.001):
        # test for sequence labeling
        good_data = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [0., 0.], [1., 1.], [2., 2.]]); # shape(6, 2)
        good_label = torch.tensor([0, 1, 2, 0, 1, 2])
        good_loss = translation_cl_loss(good_data, good_label, temperature=t)
        bad_data = torch.tensor([[0., 0.], [1, 1.], [2., 2.], [3., 3.], [4, 4.], [5., 5.]]); # shape(6, 2)
        bad_label = torch.tensor([0, 1, 2, 0, 1, 2])
        # bad_loss = scl_loss(bad_data, bad_label, temperature=t)
        bad_loss = translation_cl_loss(bad_data, bad_label, temperature=t)
        print("Temperature: {} Good Loss: {} Bad Loss: {}".format('%.4f' % t, good_loss / (len(good_data)),
                                                                  bad_loss / (len(good_data))))
        # loss = SupConLoss(contrast_mode='one', temperature=t)
        # good_data = torch.tensor([[[0., 0.]], [[1, 1.]], [[2., 2.]], [[0., 0.]], [[1, 1.]], [[2., 2.]]]);
        # good_label = torch.tensor([0, 1, 2, 0, 1, 2.])
        # good_loss = loss(good_data, good_label)
        # bad_data = torch.tensor([[[0., 0.]], [[1, 1.]], [[2., 2.]], [[3., 3.]], [[4, 4.]], [[5., 5.]]]);
        # bad_label = torch.tensor([0, 1, 2, 0, 1, 2.])
        # bad_loss = loss(bad_data, bad_label)
        # print("Temperature: {} Good Loss: {} Bad Loss: {}".format('%.4f' % t, good_loss / len(good_data),
        #                                                           bad_loss / len(good_data)))
    '''
    good_data = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [0., 0.], [1., 1.], [2., 2.]]);  # shape(6, 2)
    good_label = torch.tensor([0, 1, 2, 0, 1, 2])
    cluster_lable = torch.tensor([0, 0, 1, 0, 0, 1])
    good_loss = translation_cl_loss(good_data, good_label, cluster_lable)
    print("Temperature: {} Good Loss: {}".format('%.4f', good_loss / len(good_data)))
    # Temperature: %.4f Good Loss: 0.549064040184021
    # Temperature: %.4f Good Loss: 0.7254309058189392

    # Temperature: %.4f Good Loss: 0.6813392043113708
    # Temperature: %.4f Good Loss: 0.8967587351799011
