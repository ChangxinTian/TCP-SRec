import importlib
from tqdm import tqdm
from time import time

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from recbole.trainer import Trainer
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.utils import ModelType, get_gpu_usage, early_stopping, dict2str, set_color


def my_get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """

    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['model', model_file_name])
    model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config['device']

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask


def my_get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    if model_name == 'TCPSRec':
        return epochTrainer

    try:
        return getattr(importlib.import_module('recbole.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbole.trainer'), 'KGTrainer')
        elif model_type == ModelType.TRADITIONAL:
            return getattr(importlib.import_module('recbole.trainer'), 'TraditionalTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')


class epochTrainer(Trainer):
    def __init__(self, config, model):
        super(epochTrainer, self).__init__(config, model)
        self.train_stage = config['train_stage']
        self.pretrain_epochs = self.config['pretrain_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
        ) if show_progress else train_data)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # Change start
            losses = loss_func(interaction, epoch_idx)
            # Change end

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # Change start
            if self.train_stage == 'pretrain' and epoch_idx < self.pretrain_epochs:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Pretraining! Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            # Change end

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(valid_score,
                                                                                              self.best_valid_score,
                                                                                              self.cur_step,
                                                                                              max_step=self.stopping_step,
                                                                                              bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue') +
                                      ": %.2fs, " + set_color("valid_score", 'blue') +
                                      ": %f]") % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result


def groupby_mean_nondeterministically(value: torch.Tensor, labels: torch.LongTensor, device):
    """Group-wise average for (sparse) grouped tensors

    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)

    Returns:
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)

    Examples:
        >>> samples = torch.Tensor([
                            [0.15, 0.15, 0.15],    #-> group / class 1
                            [0.2, 0.2, 0.2],    #-> group / class 3
                            [0.4, 0.4, 0.4],    #-> group / class 3
                            [0.0, 0.0, 0.0]     #-> group / class 0
                    ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)

        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.3000, 0.3000, 0.3000]])

        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels))).to(device)
    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, value)
    result = result / labels_count.float().unsqueeze(1)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist()))).to(device)
    return result, new_labels


def groupby_mean_slow(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques_sort, index_sort, count_sort = torch.unique(labels, return_counts=True, return_inverse=True)
    uniques_sort, index_sort, count_sort = uniques_sort.tolist(), index_sort.tolist(), count_sort.tolist()
    index2items = {index: [] for index in range(len(uniques_sort))}
    for i, index in enumerate(index_sort):
        index2items[index].append(i)

    uniques_mean = torch.zeros((len(uniques_sort), value.shape[1]), dtype=value.dtype).to(device)
    for i in range(len(uniques_sort)):
        uniques_mean[i, :] = torch.mean(value[index2items[i]], dim=0)

    return uniques_mean, uniques_sort


def groupby_mean_normal(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques_sort, index_sort = torch.unique(labels, return_inverse=True)
    uniques_size = uniques_sort.shape[0]

    index2items = {index: [] for index in range(uniques_size)}
    for index in range(uniques_size):
        index2items[index] = (index_sort == index).nonzero(as_tuple=False)

    uniques_mean = torch.zeros((uniques_size, value.shape[1]), dtype=value.dtype).to(device)
    for i in range(len(uniques_sort)):
        uniques_mean[i, :] = torch.mean(value[index2items[i]], dim=0)

    return uniques_mean, uniques_sort


def groupby_mean_matrix(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    remap_labels = torch.LongTensor(list(map(key_val.get, labels))).to(device)
    M = torch.zeros(len(uniques), len(value)).to(device)
    M[remap_labels, torch.arange(len(value))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    result_mean = torch.mm(M, value)
    result_label = torch.LongTensor(list(map(val_key.get, range(len(uniques))))).to(device)

    return result_mean, result_label


def groupby_mean_sparse(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    remap_labels = torch.LongTensor(list(map(key_val.get, labels)))

    indices = torch.stack([remap_labels, torch.arange(len(value))], dim=0).to(device)
    values = torch.ones_like(remap_labels, dtype=torch.float32).to(device)
    M = torch.sparse.FloatTensor(indices, values, (len(uniques), len(value)))

    diags = torch.sparse.sum(M, dim=1).to_dense()
    diags = torch.pow(diags, -1)
    diags_lookup = diags[remap_labels]
    norm_M = torch.sparse.FloatTensor(indices, diags_lookup, (len(uniques), len(value)))

    result_mean = torch.sparse.mm(norm_M, value)
    result_label = torch.LongTensor(list(map(val_key.get, range(len(uniques))))).to(device)
    return result_mean, result_label


def groupby_mean(value: torch.Tensor, labels: torch.LongTensor, device):
    # result_mean, result_label = groupby_mean_nondeterministically(value, labels, device)
    # result_mean, result_label = groupby_mean_slow(value, labels, device)
    # result_mean, result_label = groupby_mean_normal(value, labels, device)
    # result_mean, result_label = groupby_mean_matrix(value, labels, device)
    result_mean, result_label = groupby_mean_sparse(value, labels, device)
    return result_mean, result_label
