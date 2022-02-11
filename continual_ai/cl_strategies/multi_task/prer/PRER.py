import itertools
from copy import deepcopy

import numpy as np
import os

import logging

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim, autograd
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm import tqdm

from base import Encoder, Decoder
from continual_ai.cl_strategies import NaiveMethod, Container
from .rnvp import RNVP
from .utils import Conditioner, Prior, reconstruction_loss, NoneConditioner
from continual_ai.utils import ExperimentConfig
from continual_ai.cl_settings import SingleIncrementalTaskSolver
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, \
    BatchSampler


class PRERproj(NaiveMethod):

    def __init__(self, encoder: Encoder, config: ExperimentConfig, classes: int,
                 device,
                 plot_dir: str,
                 logger: logging.Logger = None, **kwargs):
        super().__init__()

        self.config = config
        self.device = device
        self.plot_dir = plot_dir
        self.classes = classes
        self.emb_dim = encoder.embedding_dim
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(encoder.flat_cnn_size, encoder.embedding_dim),
            # torch.nn.Sigmoid()
        ).to(self.device)

        self.prior = Prior(encoder.embedding_dim)
        self.reset_nf = config.cl_technique_config.get('reset_nf', False)
        self.reset_decoder = config.cl_technique_config.get('reset_decoder', False)

        ae_config = config.cl_technique_config['autoencoder']

        self.autoencoder_lr = ae_config.get('lr', 1e-3)
        self.ae_batch_size = ae_config.get('batch_size',
                                           config.train_config['batch_size'])
        self.rec_loss_type = ae_config.get('rec_loss', 'gaussian')
        self.softclip = ae_config.get('softclip', 0)

        ae_cond = ae_config.get('conditioning_config', {'type': None})
        ae_conditioner_type = ae_cond['type']

        self.ae_conditioner = NoneConditioner()
        self.autoencoder_classifier = None

        if ae_conditioner_type == 'one_hot' or ae_conditioner_type == 'class_embedding':
            self.ae_conditioner = Conditioner(ae_cond, classes)
        # elif ae_conditioner_type == 'nn':
        #     self.autoencoder_classifier = SingleIncrementalTaskSolver(self.emb_dim, device=self.device)
        elif ae_conditioner_type == 'none':
            pass
        else:
            assert False

        self.conditioner_fine_tuning = ae_cond.get('conditioner_fine_tuning',
                                                   False) and self.autoencoder_classifier is not None

        decoder = Decoder(config.cl_config['dataset'], encoder,
                          cond_size=self.ae_conditioner.size)
        self.decoder = decoder

        ws = config.cl_technique_config.get('weights', {})

        self.old_w = ws.get('old_tasks', 0.5)
        self.er = ws.get('er', 1)
        self.l1 = ws.get('l1', 0)
        self.ce = ws.get('ce', 1)
        self.rec = ws.get('rec', 1)

        self.cl_problem = config.cl_config['cl_problem']
        self.autoencoder_epochs = config.cl_technique_config[
            'autoencoder_epochs']
        self.generator_epochs = config.cl_technique_config['generator_epochs']

        to_plot = config.cl_technique_config.get('plot', False)
        if not to_plot:
            self.plot_dir = None
        self.plot_step = config.cl_technique_config.get('plot_step', 5)

        regularization = config.cl_technique_config['regularization']
        self.fixed_replay = regularization['fixed_replay']
        assert self.fixed_replay > 0
        if self.fixed_replay < self.ae_batch_size:
            self.fixed_replay = self.ae_batch_size

        self.distance = regularization.get('distance', 'cosine')
        self.er_sample = regularization.get('to_sample', self.ae_batch_size)

        self.reg_type = regularization['type']
        assert self.reg_type in ['alternation', 'overwrite'], \
            'Regularization type can be [alternation, overwrite]'

        nf = config.cl_technique_config['nf']
        self.levels = nf.get('levels', 1)
        self.blocks = nf.get('blocks', 5)
        self.n_hidden = nf.get('n_hiddens', 1)
        self.hidden_size = nf.get('hidden_size', 1.0)
        self.nf_lr = nf.get('lr', 1e-3)
        self.nf_weight_decay = nf.get('weight_decay', 1e-5)
        self.tolerance_training = nf.get('tolerance_training', False)
        self.tolerance = nf.get('tolerance', 10)
        self.tolerance_cap = nf.get('tolerance_cap', np.inf)
        self.nf_batch_size = nf.get('batch_size',
                                    config.train_config['batch_size'])
        self.anneal_lr = nf.get('annealing', False)

        nf_cond_config = config.cl_technique_config['nf']['conditioning_config']
        conditioner_type = nf_cond_config['type']

        self.nf_conditioner = NoneConditioner()

        if conditioner_type in ['one_hot', 'class_embedding']:
            if ae_conditioner_type == conditioner_type:
                self.nf_conditioner = self.ae_conditioner
            else:
                self.nf_conditioner = Conditioner(nf_cond_config, classes)
        elif conditioner_type == 'none':
            pass
        else:
            assert False

        self.nf_conditioner_type = conditioner_type

        opt = config.train_config['optimizer']

        if opt == 'adam':
            self.autoencoder_optimizer = optim.Adam
        elif opt == 'sgd':
            self.autoencoder_optimizer = optim.SGD
        else:
            assert False

        total_params = sum(p.numel() for p in itertools.chain(
            itertools.chain(self.get_generator().parameters(),
                            self.prior.parameters(),
                            self.decoder.parameters(),
                            self.projector.parameters(),
                            self.nf_conditioner.parameters())))

        if self.autoencoder_classifier is not None:
            autoencoder_classifier = SingleIncrementalTaskSolver(self.emb_dim)
            autoencoder_classifier.add_task(classes)
            total_params += sum([p.numel() for p in
                                 autoencoder_classifier.trainable_parameters()])

        if logger is not None:
            logger.info('Offline NF parameters:')
            logger.info(F'\tAE train epochs: {self.autoencoder_epochs}')
            logger.info(F'\tNF train epochs: {self.generator_epochs}')
            logger.info(F'\tweights: {ws}')
            logger.info(F'\tNumber of parameters: {total_params}')
            logger.info(
                F'\tNF parameters: {sum(p.numel() for p in self.get_generator().parameters())}')
            logger.info(F'\tEmbedding dim: {self.emb_dim}')

        self._d2t = torch.empty(classes, device=device, dtype=torch.long).fill_(
            -1)
        self._t2d = {}

        self.old_dataset_labels = []
        self.dataset_labels = []
        self.task_labels = []
        self.old_datasets = []

        self.sampled_images_memory = None
        self.generator = self.get_generator()
        self.past_encoder = None

    def rec_loss(self, x, xr, reduction='mean'):
        if self.rec_loss_type == 'mse':
            return torch.nn.functional.mse_loss(x, xr, reduction=reduction)
        elif self.rec_loss_type == 'gaussian':
            return reconstruction_loss(x, xr, reduction=reduction,
                                       softclip=self.softclip)
        if self.rec_loss_type == 'bce':
            return torch.nn.functional.binary_cross_entropy(x, xr,
                                                            reduction=reduction)
        else:
            assert False

    def distance_function(self, e1, e2):
        if self.distance == 'cosine':
            # torch.nn.functional.normalize()
            return torch.nn.functional.cosine_similarity(e1, e2, dim=-1)
        elif self.distance == 'euclidean':
            return torch.norm(e1 - e2, p=2, dim=-1)
        else:
            assert False

    def before_gradient_calculation(self, container: Container):
        if container.current_task.index > 0:
            with torch.no_grad():

                # old_images, old_labels, _ = self.sample(self.er_sample,
                #                                         labels=self.
                #                                         old_dataset_labels)

                # old_images, old_labels = BatchSampler(
                #     ConcatDataset(self.old_datasets), self.er_sample, False)
                old_images, old_labels = next(iter(
                    DataLoader(ConcatDataset(self.old_datasets),
                               batch_size=self.er_sample, shuffle=True)))

                if container.current_epoch == 0 and \
                        not os.path.exists(
                            os.path.join(self.plot_dir, 'images_reg.png')):

                    f = torchvision.utils.make_grid(old_images.cpu(),
                                                    scale_each=True,
                                                    range=(0, 1)).numpy()
                    f = plt.imshow(np.transpose(f, (1, 2, 0)),
                                   interpolation='nearest')
                    f.figure.savefig(
                        os.path.join(self.plot_dir, 'images_reg.png'))

                # old_images, old_labels, _, classification_embeddings = \
                #     self.get_sampled_images(
                #         self.er_sample * container.current_task.index)
            # print(old_images.min(), old_images.max())
            # dis_reg = self.distance_function(container.encoder(old_images),
            #                                  classification_embeddings)
            dis_reg = 0

            for old_dataset in self.old_datasets:
                old_images, old_labels = next(iter(
                    DataLoader(old_dataset,
                               batch_size=self.er_sample, shuffle=True)))

                d = self.distance_function(container.encoder(old_images),
                                                 self.past_encoder(old_images))
                d = d.mean()
                dis_reg += d

            dis_reg /= len(self.old_datasets)

            container.current_loss += (dis_reg * self.er)

        if self.l1 > 0:
            l1_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)

            for m in itertools.chain(container.encoder.parameters()):
                l1_loss += self.l1 * m.abs().sum()

            container.current_loss += l1_loss

    def autoencoder_training(self, container):

        posfix = {}

        labels = container.current_task.dataset_labels
        labels_proportions = len(container.current_task) // len(labels)

        images, labels = [], []
        for _, im, lb in container.current_task(self.ae_batch_size):
            images.append(im)
            labels.append(lb)

        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0).long()

        dataset = TensorDataset(images, labels)

        # with torch.no_grad():
        #     if len(self.old_dataset_labels) > 0:
        #     old_images, old_labels = [], []
        #     for i in self.old_dataset_labels:
        #         images, y, _ = self.sample(labels_proportions, labels=[i])
        #         old_images.append(images)
        #         old_labels.append(y)
        #         assert not torch.isnan(images).any()
        #
        #     old_images = torch.cat(old_images, 0)
        #     old_labels = torch.cat(old_labels, 0).long()
        #
        #     print(old_images.max(), old_images.min())

        # old_dataset = TensorDataset(old_images, old_labels)

        if len(self.old_datasets) == 0:
            d = [dataset] + self.old_datasets
            dataset = ConcatDataset(d)
        # else:
        #     dataset = self.old_datasets[0]

        # dataset = container.current_task(self.ae_batch_size)
        dataset = DataLoader(dataset, batch_size=self.ae_batch_size,
                             shuffle=True)
        # else:
        #     dataset = container.current_task(self.ae_batch_size)

        """
        
        with torch.no_grad():
            for i, (_, images, y) in enumerate(
                    container.current_task(self.ae_batch_size)):
                emb = container.encoder.flatten_cnn(images)
                emb = self.projector(emb)
                embs.append(emb)
                labels.append(y)
            d = len(embs)
            
            if len(self.old_dataset_labels) > 0:
                for i in self.old_dataset_labels:
                    images, y, _ = self.sample(d, labels=[i])
                    _embs = container.encoder.flatten_cnn(images)
                    _embs = self.projector(_embs)
                    # for _emb, _y in zip(_embs, y):
                    embs.append(_embs)
                    labels.append(y)
                    # im, lb, nfemb = self.sample(d, labels=[i])
                    # emb = container.encoder.flatten_cnn(im)
                    # emb = self.projector(emb)
                    #
                    # embs.extend(emb)
                    # labels.append(lb)

            embs = torch.cat(embs, 0)
            labels = torch.cat(labels, 0).long()

            if self.sampled_images_memory is not None:
                cat = torch.cat((embs, self.sampled_images_memory.tensors[2].to(
                    self.device)), 0)
                mn = cat.mean(0)
                std = cat.std(0)
            else:
                mn = embs.mean(0)
                std = embs.std(0)

        # self.generator.set_mean_std(mn, std)

        data_to_iter = DataLoader(TensorDataset(embs, labels),
                                  batch_size=self.nf_batch_size, shuffle=True)
        """

        if self.reset_decoder:
            decoder = Decoder(self.config.cl_config['dataset'],
                              container.encoder,
                              cond_size=self.ae_conditioner.size)
            self.decoder = decoder.to(self.device)
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(container.encoder.flat_cnn_size,
                                container.encoder.embedding_dim),
                # torch.nn.Sigmoid()
            ).to(self.device)

        autoencoder_opt = Adam(itertools.chain(self.decoder.parameters(),
                                               self.ae_conditioner.parameters(),
                                               self.projector.parameters()),
                               lr=1e-3)

        x_plot, y_plot = None, None
        first_plotted = False

        container.current_task.set_labels_type('dataset')
        container.current_task.train()

        if self.tolerance_training:
            it = itertools.count(start=0, step=1)
            assert self.tolerance > 0
            ctol = self.tolerance
        else:
            ctol = -1
            it = range(self.autoencoder_epochs)

        best_loss = np.inf
        best_model = (None, None)

        ae_bar = tqdm(it, desc='AE training', leave=False)
        for e in ae_bar:
            if self.plot_dir is not None and (
                    e % self.plot_step == 0 or e + 1 == self.autoencoder_epochs):
                if x_plot is not None:
                    f = self.plot_rec(x_plot, container, y_plot)
                    f.savefig(
                        os.path.join(self.plot_dir, 'images_rec_task{}_{}.png'
                                     .format(container.current_task.index,
                                             e if not e + 1 == self.autoencoder_epochs else 'final')))
            tot_loss = []

            for i, (images, labels) in enumerate(dataset):
                # images = images.float()

                # if i == e == 0:
                #     print(images.max(), images.min())

                self.decoder.train()
                self.ae_conditioner.train()
                container.encoder.train()
                self.projector.train()

                # if container.current_task.index > 0 and self.reg_type != 'alternation':
                #     with torch.no_grad():
                #         old_images, old_labels, _, _ = self.get_sampled_images(
                #             images.size(0))
                #         images, labels = self.combine_batches(images,
                #                                               old_images,
                #                                               labels,
                #                                               old_labels)
                # f = torchvision.utils.make_grid(images.cpu(),
                #                                 scale_each=True,
                #                                 range=(0, 1)).numpy()
                # f = plt.imshow(np.transpose(f, (1, 2, 0)),
                #                interpolation='nearest')
                # plt.show()

                if i == 0 and e == 0:
                    x_plot, y_plot = images, labels
                    f = torchvision.utils.make_grid(x_plot.cpu(),
                                                    scale_each=False,
                                                    range=(0, 1)).numpy()
                    f = plt.imshow(np.transpose(f, (1, 2, 0)),
                                   interpolation='nearest')
                    f.figure.savefig(
                        os.path.join(self.plot_dir, 'images_task{}.png'
                                     .format(container.current_task.index)))

                emb = container.encoder.flatten_cnn(images)
                emb = self.projector(emb)

                x_rec = self.decoder(emb, y=self.ae_conditioner(labels.long()))
                rec_loss = self.rec_loss(x_rec, images, reduction='none')

                tot_loss.extend(rec_loss.tolist())

                loss = rec_loss.mean()

                autoencoder_opt.zero_grad()
                loss.backward()
                autoencoder_opt.step()

                ae_bar.set_postfix(posfix)

            tot_loss = np.mean(tot_loss)
            if tot_loss < best_loss:
                best_loss = tot_loss
                best_model = (self.decoder.state_dict(),
                              self.projector.state_dict())
                ctol = self.tolerance
            else:
                ctol -= 1
                if ctol <= 0 and self.tolerance_training:
                    break

            posfix.update({'tot_loss': tot_loss, 'tolerance': ctol})

        self.decoder.load_state_dict(best_model[0])
        self.projector.load_state_dict(best_model[1])

    def nf_training(self, container: Container):
        container.current_task.train()
        container.current_task.set_labels_type('dataset')
        container.encoder.eval()
        self.nf_conditioner.eval()
        self.ae_conditioner.eval()

        embs = []
        labels = []

        with torch.no_grad():
            for i, (_, images, y) in enumerate(
                    container.current_task(self.ae_batch_size)):
                emb = container.encoder.flatten_cnn(images)
                emb = self.projector(emb)
                embs.append(emb)
                labels.append(y.long())

            for old_dataset in self.old_datasets:
                # if len(self.old_dataset_labels) > 0:
                d = len(embs)
                for images, y in DataLoader(old_dataset,
                                  batch_size=self.nf_batch_size, shuffle=True):
                    # images, y, _ = self.sample(d, labels=[i])
                    _embs = container.encoder.flatten_cnn(images)
                    _embs = self.projector(_embs)
                    # for _emb, _y in zip(_embs, y):
                    embs.append(_embs)
                    labels.append(y)
                    # im, lb, nfemb = self.sample(d, labels=[i])
                    # emb = container.encoder.flatten_cnn(im)
                    # emb = self.projector(emb)
                    #
                    # embs.extend(emb)
                    # labels.append(lb)

            embs = torch.cat(embs, 0)
            labels = torch.cat(labels, 0)

            # if self.sampled_images_memory is not None:
            #     cat = torch.cat((embs, self.sampled_images_memory.tensors[2].to(
            #         self.device)), 0)
            #     mn = cat.mean(0)
            #     std = cat.std(0)
            # else:
            mn = embs.mean(0).clone().detach()
            std = embs.std(0).clone().detach()

            self.generator.set_mean_std(mn, std)
        #
        data_to_iter = DataLoader(TensorDataset(embs, labels),
                                  batch_size=self.nf_batch_size, shuffle=True)

        if self.reset_nf:
            self.generator = self.get_generator()

        inn_optimizer = torch.optim.Adam(
            itertools.chain(self.generator.parameters(),
                            self.nf_conditioner.parameters()),
            lr=self.nf_lr,
            weight_decay=self.nf_weight_decay)

        best_model_dict = (None, None)
        best_loss = np.inf
        scheduler = None

        if self.anneal_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                inn_optimizer, mode='min', factor=0.8,
                patience=self.tolerance // 3 if self.tolerance_training
                else self.generator_epochs // 10,
                verbose=False,
                threshold=0.01, threshold_mode='rel', cooldown=0,
                min_lr=0.00001,
                eps=1e-08)

        u_to_sample = self.prior.sample(100).to(self.device)

        labels = self.dataset_labels
        probs = torch.zeros(max(labels) + 1, device=self.device)
        for i in labels:
            probs[i] = 1

        m = Categorical(probs)
        y_to_sample = m.sample(torch.Size([u_to_sample.size(0)])).long()

        if self.tolerance_training:
            it = itertools.count(start=0, step=1)
            assert self.tolerance > 0
            ctol = self.tolerance
        else:
            ctol = -1
            it = range(self.generator_epochs)

        # data_to_iter1 = DataLoader(TensorDataset(embs, labels), batch_size=self.nf_batch_size * 4, shuffle=True)
        # u, log_det = self.generator(emb, y=self.nf_conditioner(labels))
        # emb, labels = self.combine_batches(emb, old_nf_emb, labels, old_labels)

        nf_bar = tqdm(it, desc='NF training', leave=False)
        for e in nf_bar:

            if self.tolerance_training and e > self.tolerance_cap:
                break

            if self.plot_dir is not None \
                    and (e % self.plot_step == 0 or (
                    e + 1 == self.generator_epochs and not self.tolerance_training)):
                f = self.plot(u=u_to_sample, y=y_to_sample)
                f.savefig(os.path.join(self.plot_dir, 'sampled_task{}_{}.png'
                                       .format(container.current_task.index,
                                               e)))

            losses = []

            for i, (emb, labels) in enumerate(data_to_iter):
                # with torch.no_grad():
                #     if container.current_task.index > 0:
                #         images, old_labels, _, _ = self.get_sampled_images(
                #             emb.size(0))
                #         old_nf_emb = container.encoder.flatten_cnn(images)
                #         old_nf_emb = self.projector(old_nf_emb)
                #         emb, labels = self.combine_batches(emb, old_nf_emb,
                #                                            labels, old_labels)
                container.encoder.eval()
                self.prior.train()
                self.nf_conditioner.train()

                u, log_det = self.generator(emb, y=self.nf_conditioner(labels))

                if len(log_det.shape) > 1:
                    log_det = log_det.sum(1)

                log_prob = self.prior.log_prob(u)
                log_prob = torch.flatten(log_prob, 1).sum(1)

                loss = -(log_prob + log_det)

                losses.extend(loss.tolist())

                loss = torch.mean(loss)

                inn_optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                 #                              25)
                inn_optimizer.step()

            mean_loss = np.mean(losses)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model_dict = (self.generator.state_dict(),
                                   self.nf_conditioner.state_dict())
                ctol = self.tolerance
            else:
                ctol -= 1
                if ctol <= 0 and self.tolerance_training:
                    break

            nf_bar.set_postfix(
                {'ctol': ctol, 'best_loss': best_loss, 'loss': mean_loss})

            if scheduler is not None:
                scheduler.step(mean_loss)

        self.generator.load_state_dict(best_model_dict[0])
        self.nf_conditioner.load_state_dict(best_model_dict[1])

        if self.plot_dir is not None:
            f = self.plot(u=u_to_sample)
            f.savefig(os.path.join(self.plot_dir,
                                   'best_model_task{}.png'.format(
                                       container.current_task.index)))

        container.current_task.set_labels_type('task')

    def combine_batches(self, xa, xb, ya, yb, zero_prob: float = None):
        assert xa.size() == xb.size(), '{} <> {}'.format(xa.size(), xb.size())

        if zero_prob is None:
            zero_prob = 1 - len(self.old_dataset_labels) / (
                len(self.dataset_labels))

        batch_size = xa.size(0)
        mask = torch.distributions.binomial.Binomial(probs=zero_prob).sample(
            (batch_size,)).to(self.device)

        x_mask = mask.clone()
        for i in range(len(xa.shape[1:])):
            x_mask.unsqueeze_(-1)

        x = xa * x_mask + xb * (1 - x_mask)
        y = ya * mask + yb.long() * (1 - mask)
        y = y.long()

        return x, y

    def sample(self, size, labels=None, y=None, u=None):

        self.generator.eval()
        self.prior.eval()
        self.decoder.eval()
        self.ae_conditioner.eval()
        self.nf_conditioner.eval()

        if u is None:
            u = self.prior.sample(size)

        if isinstance(labels, torch.Tensor):
            y = labels

        if y is None and labels is not None:
            probs = torch.zeros(max(labels) + 1, device=self.device)
            for i in labels:
                probs[i] = 1

            m = Categorical(probs)
            y = m.sample(torch.Size([u.size(0)]))
            y = y.long()

        embs, _ = self.generator.backward(u, y=self.nf_conditioner(y))

        x = self.decoder(embs, y=self.ae_conditioner(y))

        # assert y is not None
        return x, y, embs

    @torch.no_grad()
    def generate_images_memory(self, encoder):

        fixed_replay = self.fixed_replay

        images = []
        labels = []
        nf_embeddings = []
        classification_embeddings = []

        while fixed_replay > 0:
            im, lb, nfemb = self.sample(min(self.nf_batch_size, fixed_replay),
                                        labels=self.old_dataset_labels)
            cemb = encoder(im)

            images.append(im.cpu())
            labels.append(lb.cpu())
            nf_embeddings.append(nfemb.cpu())
            classification_embeddings.append(cemb.cpu())

            fixed_replay -= self.nf_batch_size

        images, labels, nf_embeddings, classification_embeddings = \
            torch.cat(images), torch.cat(labels), torch.cat(
                nf_embeddings), torch.cat(classification_embeddings)

        self.sampled_images_memory = TensorDataset(images, labels,
                                                   nf_embeddings,
                                                   classification_embeddings)

    # @torch.no_grad()
    def get_sampled_images(self, size: int):
        if size is None:
            size = len(self.sampled_images_memory)

        indices = torch.randperm(len(self.sampled_images_memory))[:size]

        images, labels, nf_embeddings, encoder_embeddings = \
            self.sampled_images_memory[indices]

        return images.to(self.device), labels.to(self.device), \
               nf_embeddings.to(self.device), encoder_embeddings.to(self.device)

    def on_task_starts(self, container: Container, *args, **kwargs):

        self.sampled_images_memory = None
        self.dataset_labels.extend(container.current_task.dataset_labels)

        dl = np.zeros(len(container.current_task.dataset_labels), dtype=int)

        for i, j in container.current_task.t2d.items():
            dl[i] = j

        dl = torch.tensor(dl, device=self.device, dtype=torch.long)

        self._t2d[container.current_task.index] = dl

        for k, v in container.current_task.d2t.items():
            self._d2t[k] = torch.tensor(v, dtype=torch.long, device=self.device)

        # container.encoder.classification()
        if container.current_task.index > 0:
            self.generate_images_memory(container.encoder)

    def on_task_ends(self, container: Container, *args, **kwargs):

        plt.close('all')

        # with autograd.detect_anomaly():
            #     pass
        self.autoencoder_training(container)
        self.nf_training(container)

        with torch.no_grad():
            labels = container.current_task.dataset_labels
            labels_proportions = len(container.current_task) // len(labels)

            for i in container.current_task.dataset_labels:
                old_images, old_labels = [], []
                images, y, _ = self.sample(labels_proportions, labels=[i])

                # nans = torch.isnan(images).sum()
                indexes = torch.any(torch.flatten(torch.isnan(images), 1), 1)
                nans = indexes.sum()
                if nans > 0:
                    print('Nans', nans)
                    assert nans < (labels_proportions * 0.5), nans
                    images = images[~indexes]
                    y = y[~indexes]

                old_images.append(images)
                old_labels.append(y)

                old_images = torch.cat(old_images, 0)
                old_labels = torch.cat(old_labels, 0).long()
                old_images[old_images < 1e-10] = 0

                old_dataset = TensorDataset(old_images, old_labels)
                self.old_datasets.append(old_dataset)

        self.old_dataset_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)
        self.sampled_images_memory = None
        self.past_encoder = deepcopy(container.encoder)

    def plot(self, labels=None, size=100, y=None, u=None):
        if labels is None:
            labels = self.dataset_labels

        labels.sort()

        if u is not None:
            size = u.size(0)

        with torch.no_grad():
            probs = torch.zeros(max(labels) + 1, device=self.device)
            for i in labels:
                probs[i] = 1

            if y is None:
                m = Categorical(probs)
                y = m.sample(torch.Size([u.size(0)]))
                y = y.long()

            x, _, _ = self.sample(size, y=y, u=u)

            f = torchvision.utils.make_grid(x.cpu(), scale_each=False,
                                            range=(0, 1)).numpy()
            f = plt.imshow(np.transpose(f, (1, 2, 0)), interpolation='nearest')

            return f.figure

    def plot_rec(self, x, container, y=None):
        container.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            emb = container.encoder.flatten_cnn(x)
            emb = self.projector(emb)
            x_rec = self.decoder(emb, y=self.ae_conditioner(y.long()))

        f = torchvision.utils.make_grid(x_rec.cpu(), scale_each=False,
                                        range=(0, 1)).numpy()
        f = plt.imshow(np.transpose(f, (1, 2, 0)), interpolation='nearest')

        return f.figure

    def get_generator(self):
        cond_size = self.nf_conditioner.size

        gen = RNVP(n_levels=self.levels, levels_blocks=self.blocks,
                   input_dim=self.emb_dim,
                   n_hidden=self.n_hidden, conditioning_size=cond_size,
                   hidden_size=self.hidden_size)

        return gen.to(self.device)
