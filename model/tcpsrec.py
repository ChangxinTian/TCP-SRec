import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

import utils


class TCPSRec(utils.SequentialRecommender):
    def __init__(self, config, dataset):
        super(TCPSRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load TCPSRec info
        self.train_stage = config['train_stage']  # pretrain or finetune
        self.pre_model_path = config['pre_model_path']  # We need this for finetune
        self.pretrain_epochs = config['pretrain_epochs']

        self.emb_CL = config['emb_CL']  # 'in' or 'out'
        self.tau = config['tau']
        self.weight = config['weight']
        self.weight_item_CL_global = config['weight']
        self.weight_item_CL_local = config['weight']
        self.weight_subseq_CL_global = config['weight']
        self.weight_subseq_CL_local = config['weight']
        self.date_preprocess(config, dataset)

        # define layers and loss, modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers,
                                              n_heads=self.n_heads,
                                              hidden_size=self.hidden_size,
                                              inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act,
                                              layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # modules for finetune
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ['pretrain', 'finetune', 'multitask']
        if self.train_stage == 'pretrain':
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            if self.pre_model_path:
                pretrained = torch.load(self.pre_model_path)
                self.logger.info(f'Load pretrained model from {self.pre_model_path}')
                self.load_state_dict(pretrained['state_dict'])
            else:
                self.apply(self._init_weights)

    def date_preprocess(self, config, dataset):
        self.TIME = config['TIME_FIELD']
        self.TIME_SEQ = self.TIME + config['LIST_SUFFIX']
        self.SESSION_ID = config['SESSION_ID_FIELD'] if config['SESSION_ID_FIELD'] else None
        self.SESSION_SEQ = config['SESSION_ID_FIELD'] + config['LIST_SUFFIX'] if config['SESSION_ID_FIELD'] else None

        # user_id:token	item_id:token	rating:float	timestamp:float	tm_hour:float	session_id:token	tm_wday:float	tm_mon:float	tm_year:float
        self.WDAY = config['WDAY_FIELD']
        self.WDAY_SEQ = self.WDAY + config['LIST_SUFFIX']
        wday_seq = dataset[self.WDAY_SEQ]

        self.HOUR = config['HOUR_FIELD']
        self.HOUR_SEQ = self.HOUR + config['LIST_SUFFIX']
        hour_seq = dataset[self.HOUR_SEQ]

        session_seq = dataset[self.SESSION_SEQ]
        unique_session, unique_counts = torch.unique_consecutive(session_seq[:, 0], return_counts=True)
        full_session_idx = torch.cumsum(unique_counts, dim=0) - 1

        self.n_sessions = dataset.num(self.SESSION_ID)
        self.session_feature_dim = 3
        session_feature = torch.zeros([self.n_sessions, self.session_feature_dim], dtype=torch.int64)
        for i in full_session_idx:
            full_session = session_seq[i]
            output, counts = torch.unique_consecutive(full_session, return_counts=True)
            cum_counts = torch.cumsum(counts, dim=0) - 1

            full_wday = wday_seq[i]
            full_hour = hour_seq[i]
            for session_id, c in zip(output, cum_counts):
                session_feature[session_id][0] = i
                session_feature[session_id][1] = full_wday[c]
                session_feature[session_id][2] = full_hour[c] // 6  # can be change
        self.session_feature = session_feature

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B H]

    def ssl_loss(self, anchor_embedding, positive_embedding, negative_embedding=None, all_embedding=None):
        if all_embedding is None:
            all_embedding = torch.cat((positive_embedding, negative_embedding), 0)

        norm_anchor_embedding = F.normalize(anchor_embedding)
        norm_positive_embedding = F.normalize(positive_embedding)
        norm_all_embedding = F.normalize(all_embedding)

        pos_score = torch.mul(norm_anchor_embedding, norm_positive_embedding).sum(dim=1)
        ttl_score = torch.matmul(norm_anchor_embedding, norm_all_embedding.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def get_mean_of_item(self, item_embs, session_seq):
        item_embs = item_embs.reshape(-1, item_embs.shape[-1])
        session_seq = session_seq.reshape(-1)
        session_mean, session_labels = utils.groupby_mean(item_embs, session_seq, self.device)
        return session_mean, session_labels

    def item_CL_global(self, item_seq, item_embs):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        sequence_sum = torch.sum(item_embs * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])
        sequence_mean = sequence_sum / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        sequence_embs_idx = torch.nonzero(real_item_mask)[:, 0]  # torch.Size([X,])
        sequence_embs = sequence_mean[sequence_embs_idx]  # torch.Size([X, 64])

        item_CL_global_loss = self.ssl_loss(anchor_embedding=real_item_embs,
                                            positive_embedding=sequence_embs,
                                            all_embedding=sequence_mean)
        return item_CL_global_loss

    def item_CL_local(self, item_seq, item_embs, session_seq, session_mean, session_labels):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        session_label2idx = {l: i for i, l in enumerate(session_labels.tolist())}
        real_session_seq = torch.masked_select(session_seq, real_item_mask)
        session_embs_idx = [session_label2idx[i] for i in real_session_seq.tolist()]
        session_embs = session_mean[session_embs_idx]

        item_CL_local_loss = self.ssl_loss(anchor_embedding=real_item_embs,
                                           positive_embedding=session_embs,
                                           all_embedding=session_mean)
        return item_CL_local_loss

    def subseq_CL_alone(self, session_mean, session_labels):
        session_feature = self.session_feature[session_labels]

        # long periodicity
        session_feature_long = session_feature[:, 0] * 10 + session_feature[:, 1]
        cluster_mean_long, cluster_labels_long = utils.groupby_mean(session_mean, session_feature_long, self.device)

        cluster_label2idx_long = {l: i for i, l in enumerate(cluster_labels_long.tolist())}
        cluster_embs_idx_long = [cluster_label2idx_long[i] for i in session_feature_long.tolist()]
        cluster_embs_long = cluster_mean_long[cluster_embs_idx_long]

        subseq_CL_alone_loss_long = self.ssl_loss(anchor_embedding=session_mean,
                                                  positive_embedding=cluster_embs_long,
                                                  all_embedding=cluster_mean_long)

        # short periodicity
        session_feature_short = session_feature[:, 0] * 10 + session_feature[:, 2]
        cluster_mean_short, cluster_labels_long = utils.groupby_mean(session_mean, session_feature_short, self.device)

        cluster_label2idx_short = {l: i for i, l in enumerate(cluster_labels_long.tolist())}
        cluster_embs_idx_short = [cluster_label2idx_short[i] for i in session_feature_short.tolist()]
        cluster_embs_short = cluster_mean_short[cluster_embs_idx_short]

        subseq_CL_alone_loss_short = self.ssl_loss(anchor_embedding=session_mean,
                                                   positive_embedding=cluster_embs_short,
                                                   all_embedding=cluster_mean_short)

        return subseq_CL_alone_loss_long + subseq_CL_alone_loss_short

    def subseq_CL_cross(self, session_mean, session_labels):
        session_feature = self.session_feature[session_labels]

        session_feature = session_feature[:, 0] * 100 + session_feature[:, 1] * 10 + session_feature[:, 2]
        cluster_mean, cluster_labels = utils.groupby_mean(session_mean, session_feature, self.device)

        cluster_label2idx = {l: i for i, l in enumerate(cluster_labels.tolist())}
        cluster_embs_idx = [cluster_label2idx[i] for i in session_feature.tolist()]
        cluster_embs = cluster_mean[cluster_embs_idx]

        subseq_CL_cross_loss = self.ssl_loss(anchor_embedding=session_mean,
                                             positive_embedding=cluster_embs,
                                             all_embedding=cluster_mean)
        return subseq_CL_cross_loss

    def calculate_loss(self, interaction, epoch=100):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        loss = 0

        if self.train_stage == 'finetune' or self.train_stage == 'multitask' or (self.train_stage == 'pretrain'
                                                                                 and epoch >= self.pretrain_epochs):
            item_output = self.forward(item_seq, item_seq_len)
            seq_output = self.gather_indexes(item_output, item_seq_len - 1)
            pos_items = interaction[self.POS_ITEM_ID]
            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                loss += self.loss_fct(pos_score, neg_score)
            elif self.loss_type == 'CE':
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss += self.loss_fct(logits, pos_items)

        if self.train_stage == 'multitask' or (self.train_stage == 'pretrain' and epoch < self.pretrain_epochs):
            if self.emb_CL == 'in':
                item_embs = self.item_embedding(item_seq)
            elif self.emb_CL == 'out':
                item_embs = item_output if self.train_stage == 'multitask' else self.forward(
                    item_seq, item_seq_len)  # torch.Size([1024, 50, 64])

            session_seq = interaction[self.SESSION_SEQ]  # torch.Size([1024, 50])
            session_mean, session_labels = self.get_mean_of_item(item_embs, session_seq)

            item_CL_seq_loss = self.item_CL_global(item_seq, item_embs)
            loss += self.weight_item_CL_global * item_CL_seq_loss

            item_CL_subseq_loss = self.item_CL_local(item_seq, item_embs, session_seq, session_mean, session_labels)
            loss += self.weight_item_CL_local * item_CL_subseq_loss

            subseq_CL_seq_loss = self.subseq_CL_alone(session_mean, session_labels)
            loss += self.weight_subseq_CL_global * subseq_CL_seq_loss

            subseq_CL_subseq_loss = self.subseq_CL_cross(session_mean, session_labels)
            loss += self.weight_subseq_CL_local * subseq_CL_subseq_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(item_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(item_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
