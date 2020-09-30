from collections import Counter
from copy import deepcopy

import itertools
from functools import reduce
from operator import mul

import numpy as np
import os

import logging

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import optim
from torch.distributions import Categorical
from torch.nn import NLLLoss
from torch.optim import Adam, SGD

from continual_ai.cl_strategies import NaiveMethod, Container
from .custom_nf import ChannelWiseNF, get_f
from .dcgan import DCGAN
from .maf import MAF
from .rnvp import RNVP, ActNorm, BatchNorm
from .student_teacher import StudentTeacher
from .utils import Conditioner, Prior, modify_batch, generate_batch, generate_embeddings
from continual_ai.utils import ExperimentConfig
from .vae import VAE
from continual_ai.cl_settings import SingleIncrementalTaskSolver
from torch.utils.data import Dataset, DataLoader, SequentialSampler


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.fn)

    def fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


def reconstruction_loss(x_hat, x):
    log_sigma = ((x - x_hat) ** 2).mean().sqrt().log()
    log_sigma = -6 + torch.nn.functional.softplus(log_sigma + 6)
    rec = gaussian_nll(x_hat, log_sigma, x)
    # rec = 0.5 * torch.pow((x - x_hat) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    return rec.sum()


class PRER(NaiveMethod):
    # TODO: valutare segunti modifche: 1) prior per ogni task con una piccola rete che prende un embedding
    #  e genera mu e sigma, 2) Multi head generative model, 3) Miglioramento imamgini con metodi
    #  adversarial (https://arxiv.org/pdf/1704.02304.pdf), 4) Implementare samples delle Y in base all'embedding
    #  delle classi del task corrente (solo se il conditioner utilizza non one_hot) 5) annettere Prior e Conditioner nel
    #  NF (?)

    def __init__(self, decoder: torch.nn.Module, emb_dim: tuple, config: ExperimentConfig, classes: int, device,
                 plot_dir: str,
                 logger: logging.Logger = None, **kwargs):
        super().__init__()

        self.config = config
        self.device = device

        self.plot_dir = plot_dir
        # self.plot_dir = None

        self.classes = classes

        ws = {'old_tasks': 0.5, 'er': 1, 'l1': 0.0001, 'ce': 1}
        ws.update(config.cl_technique_config.get('weights', {}))

        self.generator_epochs = config.cl_technique_config.get('generator_epochs', config.train_config['epochs'])
        self.train_batch_size = config.cl_technique_config.get('train_batch_size', config.train_config['batch_size'])

        rec_loss = config.cl_technique_config.get('rec_loss', 'bce')

        if rec_loss == 'mse':
            # self.rec_loss = torch.nn.MSELoss(reduction='mean')
            self.rec_loss = torch.nn.functional.mse_loss
        elif rec_loss == 'gaussian':
            self.rec_loss = reconstruction_loss
        elif rec_loss == 'bce':
            # self.rec_loss = torch.nn.BCELoss(reduction='mean')
            self.rec_loss = torch.nn.functional.binary_cross_entropy
        else:
            assert False

        prior_std = config.cl_technique_config.get('prior_std', 1)
        assert prior_std > 0

        self.old_w = ws.get('old_tasks', 0.5)
        self.er = ws.get('er', 1)
        self.l1 = ws.get('l1', 0.001)
        self.cew = ws.get('ce', 1)

        self.cl_problem = config.cl_config['cl_problem']

        assert 0 < self.old_w < 1

        self.decoder = decoder
        self.old_decoder = deepcopy(decoder)
        self.conditioner = Conditioner(config.cl_technique_config, classes, emb_dim=emb_dim)
        self.autoencoder_classifier = None

        self.emb_dim = emb_dim

        self.autoencoder_epochs = config.cl_technique_config.get('autoencoder_epochs')

        self.model = config.cl_technique_config.get('model_type', 'rnvp').lower()
        self.n_levels = config.cl_technique_config.get('n_levels', 1)
        self.blocks = config.cl_technique_config.get('blocks', 5)
        self.n_hidden = config.cl_technique_config.get('n_hiddens', 0)
        self.hidden_size = config.cl_technique_config.get('hidden_size', 1.0)
        self.autoencoder_lr = config.train_config['lr']
        self.z_dim = config.cl_technique_config.get('z_dim', emb_dim)
        # se fixed remplay è intero allora è il numero di immagini da mettere nella memoria da cui pescare immagini
        self.fixed_replay = config.cl_technique_config.get('fixed_replay', False)

        # self.to_plot = config.cl_technique_config.get('to_plot', False)

        # if self.model == 'rnvp': self.generator = RNVP(n_levels=self.n_levels, levels_blocks=self.blocks,
        # n_hidden=self.n_hidden, input_dim=emb_dim, conditioning_size=self.conditioner.size,
        # hidden_size=self.hidden_size)

        # for module in self.generator.modules():
        #     if isinstance(module, torch.nn.Linear):
        #         torch.nn.init.orthogonal_(module.weight)
        #         if hasattr(module, 'bias') and module.bias is not None:
        #             module.bias.data.fill_(0)

        # if self.model in ['gan', 'vae']:
        #     self.prior = Prior(self.z_dim, std=prior_std)
        # else:

        if hasattr(emb_dim, '__len__'):
            self.prior = Prior(reduce(mul, emb_dim, 1))
        else:
            self.prior = Prior(emb_dim)

        # self.prior = Prior((emb_dim[0], emb_dim[1] * emb_dim[2]))

        # print(self.generator)

        # if model == 'maf':
        #     self.nf = MAF(n_blocks=self.n_levels, levels_blocks=self.blocks, cond_label_size=self.conditioner.size,
        #                    input_size=emb_dim, conditioning_size=self.conditioner.size)

        opt = config.train_config['optimizer']

        if opt == 'adam':
            self.autoencoder_optimizer = optim.Adam
        elif opt == 'sgd':
            self.autoencoder_optimizer = optim.SGD
        else:
            assert False

        self.generator = self.get_generator()
        print(self.generator)

        total_params = sum(p.numel() for p in itertools.chain(self.generator.parameters(),
                                                              self.conditioner.parameters(),
                                                                   self.prior.parameters()))

        if logger is not None:
            logger.info('Offline NF parameters:')
            logger.info(F'\tLevels: {self.n_levels}')
            logger.info(F'\tBlocks: {self.blocks}')
            logger.info(F'\tN hiddens: {self.n_hidden}')
            logger.info(F'\tHidden layers dim: {self.hidden_size}')
            logger.info(F'\tNF train epochs: {self.generator_epochs}')
            logger.info(F'\tReconstruction loss: {rec_loss}')
            logger.info(F'\tTraining batch size: {self.train_batch_size}')
            # logger.info(F'\tRegularization batch size: {self.reg_batch_size}')
            logger.info(F'\tweights: {ws}')
            logger.info(F'\tNumber of parameters: {total_params}')
            logger.info(F'\tEmbedding dim: {self.emb_dim}')

        self.autoencoder_opt = None
        self.hooks = []

        self._d2t = torch.empty(classes, device=device, dtype=torch.long).fill_(-1)

        self._t2d = {}

        # self.all_labels = []
        self.old_labels = []
        self.task_labels = []

        self.old_generator = None
        # self.old_decoder = None
        self.old_conditioner = None
        self.sampled_images_memory = None

    def autoencoder_training(self, container):

        for param in container.encoder.parameters():
            param.requires_grad = True

        _, images, labels = next(iter(container.current_task(self.train_batch_size)))

        emb = container.encoder(images)
        emb = torch.flatten(emb, 1)

        if self.autoencoder_classifier is None:
            self.autoencoder_classifier = torch.nn.Linear(emb.size(1), self.classes).to(self.device)

        autoencoder_opt = Adam(
            itertools.chain(itertools.chain(self.decoder.parameters(), self.autoencoder_classifier.parameters(),
                                            container.encoder.parameters())), lr=0.001)

        container.current_task.set_labels_type('dataset')
        container.current_task.train()

        hooks = []
        for n, m in itertools.chain(self.decoder.named_modules(), container.encoder.named_modules()):
            if isinstance(m, torch.nn.ReLU):  # or isinstance(m, torch.nn.Conv2d):
                hooks.append(Hook(m))

        _, x_plot, y_plot = container.current_task.sample(size=self.train_batch_size)

        for e in range(self.autoencoder_epochs):
            tot_sparsity = []

            if e % 5 == 0 and e > 0 and self.plot_dir is not None:
                f = self.plot_rec(x_plot, container)
                f.savefig(os.path.join(self.plot_dir, '{}_rec_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                self.decoder.train()
                container.encoder.train()

                emb = container.encoder(images)

                x_rec = self.decoder(emb)

                assert not torch.isnan(images).any(), 'Sampled Images NaN'
                assert not torch.isnan(emb).any(), 'Emb NaN'
                assert not torch.isnan(x_rec).any(), 'X_rec NaN'

                pred = self.autoencoder_classifier(torch.flatten(emb, 1))
                cross_entropy = torch.nn.functional.cross_entropy(pred,
                                                                  labels.long(), reduction='none')
                cross_entropy = cross_entropy.mean()
                cross_entropy *= self.cew

                rec_loss = self.rec_loss(x_rec, images, reduction='mean')

                l1_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                for h in hooks:
                    l1_loss += torch.abs(h.output).mean()

                loss = rec_loss + cross_entropy  # + dis_reg * self.er #+ l1_loss * self.l1

                # autoencoder_opt.zero_grad()
                # loss.backward()
                # autoencoder_opt.step()

                if container.current_task.index > 0:
                    images, labels, old_embeddings = self.get_sampled_images(emb.size(0))

                    emb = container.encoder(images)
                    x_rec = self.decoder(emb)

                    pred = self.autoencoder_classifier(torch.flatten(emb, 1))
                    cross_entropy = torch.nn.functional.cross_entropy(pred,
                                                                      labels.long(), reduction='none')
                    cross_entropy = cross_entropy.mean()
                    cross_entropy *= self.cew

                    dis_reg = torch.sub(1.0, torch.nn.functional.cosine_similarity(torch.flatten(emb, 1),
                                                                                   torch.flatten(old_embeddings, 1),
                                                                                   dim=-1))

                    dis_reg = dis_reg.mean()

                    rec_loss = self.rec_loss(x_rec, images)

                    # print(e, i, dis_reg.item())

                    old_loss = rec_loss + cross_entropy + dis_reg * self.er  # + l1_loss * self.l1

                    loss = (1 - self.old_w) * loss + self.old_w * old_loss

                    # autoencoder_opt.zero_grad()
                    # loss.backward()
                    # autoencoder_opt.step()

                    if e % 5 == 0 and e > 0 and self.plot_dir is not None:
                        f = self.plot_rec(x_rec, container)
                        f.savefig(os.path.join(self.plot_dir, '{}_oldtask_rec_images_task{}.png'
                                               .format(e, container.current_task.index)))
                        plt.close(f)

                autoencoder_opt.zero_grad()
                loss.backward()
                autoencoder_opt.step()

        for h in hooks:
            h.close()

        for param in container.encoder.parameters():
            param.requires_grad = False

    def nf_training(self, container: Container):

        # self.generator = self.get_generator()

        container.encoder.eval()

        inn_optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters(),
                                                         self.conditioner.parameters()), lr=1e-4, weight_decay=1e-5)

        container.current_task.set_labels_type('dataset')
        to_plot = list(itertools.chain(*self.task_labels, container.current_task.dataset_labels))
        
        # if False:
        #     with torch.no_grad():
        #         indexes = []
        #         losses = []
        #         self.decoder.train()
        #         container.encoder.train()
        #
        #         for _, (i, images, labels) in enumerate(container.current_task(self.train_batch_size)):
        #             emb = container.encoder(images)
        #
        #             x_rec = self.decoder(emb)
        #
        #             pred = self.autoencoder_classifier(torch.flatten(emb, 1))
        #             cross_entropy = torch.nn.functional.cross_entropy(pred,
        #                                                               labels.long(), reduction='none')
        #             cross_entropy *= self.cew
        #             rec_loss = self.rec_loss(x_rec, images, reduction='none').view(cross_entropy.size(0), -1).sum(1)
        #
        #             loss = cross_entropy + rec_loss
        #
        #             losses.extend(loss)
        #             indexes.extend(i)
        #
        #         # sort desc
        #         # prendo indici ed estraggo gli indici veri
        #         # faccio un sampling del sottoinsieme
        #
        #         losses = torch.stack(losses).tolist()
        #
        #         values = zip(losses, indexes)
        #         values = sorted(values, key=lambda x: x[0], reverse=True)
        #         losses, indexes = zip(*values)
        #         # indexes = indexes[self.subset_size]
        #
        #         indexes = indexes[5000:]
        #         # indexes = torch.tensor(indexes, device=self.device)
        #
        #         data_to_iter = container.current_task(self.train_batch_size,
        #                                               sampler=torch.utils.data.SubsetRandomSampler(indexes))
        # else:

        data_to_iter = container.current_task(self.train_batch_size)

        for e in range(self.generator_epochs):
            if e % 5 == 0 and self.plot_dir is not None:
                f = self.plot(to_plot)
                f.savefig(os.path.join(self.plot_dir, 'sampled_images_task{}_{}.png'
                                       .format(container.current_task.index, e)))
                plt.close(f)

                # with torch.no_grad():
                #     container.encoder.eval()
                #     container.solver.eval()
                #
                #     true_labels = []
                #     predicted_labels = []
                #
                #     _true_labels = []
                #     _predicted_labels = []
                #
                #     # sampled = generate_batch(1000, self.old_decoder, self.generator,
                #     #                          container.current_task.dataset_labels, self.conditioner,
                #     #                          self.prior, self.device)
                #
                #     to_sample = 100
                #     sampled = generate_embeddings(to_sample, self.generator,
                #                                   container.current_task.dataset_labels, self.conditioner,
                #                                   self.prior, self.device)
                #     sampled = tuple(zip(*sampled))
                #
                #     sampled_images_memory = DataLoader(sampled, 32, drop_last=False)
                #     tot_similarity = 0
                #
                #     for emb, y in sampled_images_memory:
                #         true_labels.extend(y.tolist())
                #         # emb = container.encoder(x)
                #         a = container.solver(emb, task=0)
                #         predicted_labels.extend(a.max(dim=1)[1].tolist())
                #
                #         new_emb = container.encoder(self.decoder(emb))
                #
                #         dis_reg = torch.nn.functional.cosine_similarity(torch.flatten(emb, 1),
                #                                                         torch.flatten(
                #                                                             new_emb, 1),
                #                                                         dim=-1)
                #         tot_similarity += dis_reg.sum().item()
                #
                #         a = container.solver(new_emb, task=0)
                #         _predicted_labels.extend(a.max(dim=1)[1].tolist())
                #
                #     xt, xp = np.asarray(true_labels), np.asarray(predicted_labels)
                #
                #     _xp = np.asarray(_predicted_labels)
                #
                #     print(e, (xt == xp).sum() / len(xt), (xt == _xp).sum() / len(xt),  tot_similarity / to_sample)
                #
                #     del sampled

            for i, (_, images, labels) in enumerate(data_to_iter):
                with torch.no_grad():
                    emb = container.encoder(images)
                    # emb = torch.nn.functional.dropout(emb, .5, True)

                self.generator.train()
                self.prior.train()
                self.conditioner.train()

                # u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))
                u, log_det = self.generator(emb)

                if len(log_det.shape) > 1:
                    log_det = log_det.sum(1)

                log_prob = self.prior.log_prob(u)
                log_prob = torch.flatten(log_prob, 1).sum(1)

                loss = -torch.mean(log_prob + log_det)

                # inn_optimizer.zero_grad()
                # loss.backward()
                # inn_optimizer.step()

                if container.current_task.index > 0:
                    with torch.no_grad():

                        images, labels, old_embeddings = self.get_sampled_images(emb.size(0))
                        emb = container.encoder(images)

                    u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))

                    if len(log_det.shape) > 1:
                        log_det = log_det.sum(1)

                    log_prob = self.prior.log_prob(u)
                    log_prob = torch.flatten(log_prob, 1).sum(1)

                    old_loss = -torch.mean(log_prob + log_det)

                    loss = (1 - self.old_w) * loss + self.old_w * old_loss

                inn_optimizer.zero_grad()
                loss.backward()
                inn_optimizer.step()

        container.current_task.set_labels_type('task')

    def generate_embeddings(self, size, generative_model, labels, conditioner, prior):
        generative_model.eval()
        conditioner.eval()
        prior.eval()

        if conditioner is not None:
            probs = torch.zeros(max(labels) + 1, device=self.device)
            for i in labels:
                probs[i] = 1

            m = Categorical(probs)
            y = m.sample(torch.Size([size]))
            y_cond = conditioner(y)
            y = y.long()
        else:
            y_cond = None
            y = None

        u = prior.sample(size)
        embs, _ = generative_model.backward(u, y=y_cond)

        return embs, y

    def generate_conditioned_batch(self, size, reconstructor, generative_model, labels, conditioner, prior):
        reconstructor.eval()

        embs, y = self.generate_embeddings(size, generative_model, labels, conditioner, prior)
        z = reconstructor(embs)

        return z, y, embs

    def generate_predicted_batch(self, size, reconstructor, generative_model, labels, predicter, prior):
        reconstructor.eval()
        predicter.eval()

        embs, _ = self.generate_embeddings(size, generative_model, labels, None, prior)
        pred = predicter(embs)
        y = pred.max(dim=1)[1]
        y = y.long()

        z = reconstructor(embs)

        return z, y, embs

    @torch.no_grad()
    def get_sampled_images(self, size: int):
        if self.fixed_replay:
            if self.sampled_images_memory is None:
                print('NONE')
                with torch.no_grad():
                    sampled = generate_batch(self.fixed_replay, self.decoder, self.generator,
                                             self.old_labels, self.conditioner, self.prior, self.device)

                sampled = tuple(zip(*sampled))

                self.sampled_images_memory = itertools.cycle(DataLoader(sampled, size, drop_last=False))

            images, labels, old_embeddings = next(self.sampled_images_memory)
            return images, labels, old_embeddings
        else:
            images, labels, old_embeddings = generate_batch(size, self.old_decoder,
                                                            self.old_generator,
                                                            self.old_labels, self.old_conditioner,
                                                            self.prior, self.device)
            return images, labels, old_embeddings

    def on_task_starts(self, container: Container, *args, **kwargs):

        self.sampled_images_memory = None

        dl = np.zeros(len(container.current_task.dataset_labels), dtype=int)

        for i, j in container.current_task.t2d.items():
            dl[i] = j

        dl = torch.tensor(dl, device=self.device, dtype=torch.long)

        for n, m in itertools.chain(self.decoder.named_modules(), container.encoder.named_modules()):
            if isinstance(m, torch.nn.ReLU):  # or isinstance(m, torch.nn.Conv2d):
                self.hooks.append(Hook(m))

        self._t2d[container.current_task.index] = dl

        for k, v in container.current_task.d2t.items():
            self._d2t[k] = torch.tensor(v, dtype=torch.long, device=self.device)

        self.autoencoder_training(container)

        for h in self.hooks:
            h.close()

    def on_task_ends(self, container: Container, *args, **kwargs):

        self.nf_training(container)

        # with torch.no_grad():
        #     print('Prova')
        #     container.encoder.eval()
        #     container.solver.eval()
        #
        #     true_labels = []
        #     predicted_labels = []
        #
        #     # sampled = generate_batch(1000, self.old_decoder, self.generator,
        #     #                          container.current_task.dataset_labels, self.conditioner,
        #     #                          self.prior, self.device)
        #     sampled = generate_embeddings(1000, self.generator,
        #                                   container.current_task.dataset_labels, self.conditioner,
        #                                   self.prior, self.device)
        #     sampled = tuple(zip(*sampled))
        #
        #     sampled_images_memory = DataLoader(sampled, 32, drop_last=False)
        #
        #     for emb, y in sampled_images_memory:
        #         true_labels.extend(y.tolist())
        #         # emb = container.encoder(x)
        #         a = container.solver(emb, task=0)
        #         predicted_labels.extend(a.max(dim=1)[1].tolist())
        #
        #     xt, xp = np.asarray(true_labels), np.asarray(predicted_labels)
        #
        #     print((xt == xp).sum() / len(xt))

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)
        self.sampled_images_memory = None

        if self.fixed_replay:
            self.old_generator = self.get_generator().to(self.device)
            self.old_generator.load_state_dict(self.generator.state_dict())
            self.old_conditioner = deepcopy(self.conditioner)
            self.old_generator.eval()
            self.old_decoder.load_state_dict(self.decoder.state_dict())
            self.old_decoder.eval()

        if self.plot_dir is not None:
            _, x_plot, _ = container.current_batch
            f = self.plot_rec(x_plot, container)
            f.savefig(os.path.join(self.plot_dir, 'rec_images_task{}_{}.png'
                                   .format(container.current_task.index, 'final')))
            plt.close(f)

        if self.plot_dir is not None:
            f = self.plot()
            f.savefig(
                os.path.join(self.plot_dir, 'sampled_images_task{}_final.png'.format(container.current_task.index)))
            plt.close(f)

    def plot(self, labels=None):
        if labels is None:
            labels = []
            for tl in self.task_labels:
                labels.extend(tl)

        labels.sort()
        sample_n = 10

        with torch.no_grad():
            self.generator.eval()
            self.decoder.eval()

            f, axs = plt.subplots(nrows=len(labels), ncols=sample_n)  # , figsize=(20, 20))

            for i, label in enumerate(labels):
                for k in range(sample_n):

                    y = torch.tensor([label], dtype=torch.long, device=self.device)
                    u = self.prior.sample(1)
                    # print(u.shape)

                    # emb, _ = self.generator.backward(u, y=self.conditioner(y))
                    emb, _ = self.generator.backward(u)

                    # emb = torch.cat([emb, self.conditioner(y)], 1)

                    z = self.decoder(emb)
                    z = z.cpu().numpy()[0]

                    mn = z.min()

                    if mn < 0:
                        z = (z + 1) / 2

                    if z.shape[0] == 1:
                        z = z[0]
                    else:
                        z = np.moveaxis(z, 0, -1)

                    if len(labels) > 1:
                        axs[i][k].imshow(z)
                    else:
                        axs[k].imshow(z)

            # if len(labels) > 1:
            for ax in axs:
                if len(labels) > 1:
                    for a in ax:
                        a.set_xticklabels([])
                        a.set_yticklabels([])
                        a.set_aspect('equal')
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')

        return f

    def plot_rec(self, x, container):

        with torch.no_grad():
            # x, labels = next(iter(container.current_task(batch_size=self.train_batch_size)))
            emb = container.encoder(x)
            # emb = torch.cat([emb, self.conditioner(labels)], 1)

            # pca_emb = PCA(n_components=2).fit_transform(emb.cpu())
            #
            # plt.scatter(pca_emb[:, 0], pca_emb[:, 1])
            # plt.show()

            x_rec = self.decoder(emb)
            x_rec = x_rec.cpu().numpy()
            x = x.cpu().numpy()

        mn = min(x_rec.min(), x.min())

        if mn < 0:
            x_rec = (x_rec + 1) / 2
            x = (x + 1) / 2

        cols = min(10, len(x))
        f, axs = plt.subplots(nrows=2, ncols=cols, figsize=(20, 20))

        if x_rec.shape[1] == 1:
            x_rec = x_rec[:, 0]
            x = x[:, 0]
        else:
            x_rec = np.moveaxis(x_rec, 1, -1)
            x = np.moveaxis(x, 1, -1)

        for c in range(cols):
            axs[0][c].imshow(x[c])
            axs[1][c].imshow(x_rec[c])

        for ax in axs:
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_aspect('equal')

        f.subplots_adjust(wspace=0, hspace=0)

        return f

    def get_generator(self):

        # cf = get_f(hidden_dim_channel=10.0, hidden_dim_width=10.0, hidden_layers_width=0,
        #            hidden_layers_channel=0,
        #            model='linear')
        # gf = get_f(hidden_dim_channel=2.0, hidden_dim_width=2.0, hidden_layers_width=0,
        #            hidden_layers_channel=0,
        #            model='linear')
        #
        # gen = ChannelWiseNF(n_levels=self.n_levels, levels_blocks=self.blocks, input_dim=self.emb_dim,
        #                     coupling_f=cf,
        #                     gaussianize_f=gf, conditioning_size=self.conditioner.size).cuda()

        # gen = MAF(self.blocks, self.emb_dim, 600, self.n_levels, self.conditioner.size,)

        # gen = RNVP(n_levels=self.n_levels, levels_blocks=self.blocks, input_dim=self.emb_dim,
        #            n_hidden=self.n_hidden, conditioning_size=self.conditioner.size, hidden_size=self.hidden_size)

        gen = RNVP(n_levels=self.n_levels, levels_blocks=self.blocks, input_dim=self.emb_dim,
                   n_hidden=self.n_hidden, conditioning_size=0, hidden_size=self.hidden_size)

        return gen.to(self.device)
