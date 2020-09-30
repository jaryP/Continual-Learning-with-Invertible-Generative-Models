from collections import Counter
from copy import deepcopy

import itertools
from functools import reduce
from operator import mul

import numpy as np
import os

import logging

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torch.distributions import Categorical
from torch.optim import Adam, SGD

from continual_ai.cl_strategies import NaiveMethod, Container
from .rnvp import RNVP, ActNorm, BatchNorm
from .utils import Conditioner, Prior, modify_batch, generate_batch, generate_embeddings
from continual_ai.utils import ExperimentConfig
from continual_ai.cl_settings import SingleIncrementalTaskSolver
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset


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
    # TODO: 1) Scelta del tipo di regolarizzazione (alternation, overwriting, concatenation), 2) pulire codice,
    #  3) regolarizzazione AE, 4) spostarsi su plot pytorch
    # TODO: 1) Provare alternation, overwriting (done), concatenation

    def __init__(self, decoder: torch.nn.Module, emb_dim: tuple, config: ExperimentConfig, classes: int, device,
                 plot_dir: str,
                 logger: logging.Logger = None, **kwargs):
        super().__init__()

        self.config = config
        self.device = device

        self.plot_dir = plot_dir

        self.classes = classes

        ws = {'old_tasks': 0.5, 'er': 1, 'l1': 0.0001, 'ce': 1}
        ws.update(config.cl_technique_config.get('weights', {}))

        self.train_batch_size = config.cl_technique_config.get('train_batch_size', config.train_config['batch_size'])

        rec_loss = config.cl_technique_config.get('rec_loss', 'bce')

        if rec_loss == 'mse':
            self.rec_loss = torch.nn.functional.mse_loss
        elif rec_loss == 'gaussian':
            self.rec_loss = reconstruction_loss
        elif rec_loss == 'bce':
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
        self.autoencoder_epochs = config.cl_technique_config['autoencoder_epochs']
        self.generator_epochs = config.cl_technique_config['generator_epochs']

        # self.model = config.cl_technique_config.get('model_type', 'rnvp').lower()
        # self.n_levels = config.cl_technique_config.get('n_levels', 1)
        # self.blocks = config.cl_technique_config.get('blocks', 5)
        # self.n_hidden = config.cl_technique_config.get('n_hiddens', 0)
        # self.hidden_size = config.cl_technique_config.get('hidden_size', 1.0)
        self.autoencoder_lr = config.train_config['lr']
        self.z_dim = config.cl_technique_config.get('z_dim', emb_dim)
        # se fixed remplay è intero allora è il numero di immagini da mettere nella memoria da cui pescare immagini
        # self.fixed_replay = config.cl_technique_config.get('fixed_replay', False)
        # self.batch_combination = config.cl_technique_config.get('batch_combination', 'overwrite')
        to_plot = config.cl_technique_config.get('plot', False)
        if not to_plot:
            self.plot_dir = None

        self.plot_step = config.cl_technique_config.get('plot_step', 5)

        self.ae_conditioned = config.cl_technique_config.get('ae_conditioned', True)
        self.conditioner_fine_tuning = config.cl_technique_config['conditioning_config'] \
            .get('conditioner_fine_tuning', False)

        regularization = config.cl_technique_config['regularization']
        self.fixed_replay = regularization['fixed_replay']
        self.reg_type = regularization['type']
        assert self.reg_type in ['concatenation', 'alternation', 'overwrite'], \
            'Regularization type can be [concatenation, alternation, overwrite]'

        nf = config.cl_technique_config['nf']
        self.levels = nf.get('levels', 1)
        self.blocks = nf.get('blocks', 5)
        self.n_hidden = nf.get('n_hiddens', 0)
        self.hidden_size = nf.get('hidden_size', 1.0)
        self.nf_lr = nf.get('lr', 1e-3)
        self.nf_weight_decay = nf.get('weight_decay', 1e-5)
        self.tolerance_training = nf.get('tolerance_training', False)
        self.tolerance = nf.get('tolerance', 10)

        assert 0 < self.old_w < 1

        self.decoder = decoder

        cod_type = config.cl_technique_config['conditioning_config']['type']

        if cod_type == 'autoencoder':
            assert self.ae_conditioned
            self.conditioner = None
        elif cod_type in ['one_hot', 'class_embedding']:
            self.conditioner = Conditioner(config.cl_technique_config, classes, emb_dim=emb_dim)
        else:
            assert False

        self.autoencoder_classifier = None

        self.emb_dim = emb_dim

        if hasattr(emb_dim, '__len__'):
            emb_dim = reduce(mul, emb_dim, 1)

        self.prior = Prior(emb_dim)

        opt = config.train_config['optimizer']

        if opt == 'adam':
            self.autoencoder_optimizer = optim.Adam
        elif opt == 'sgd':
            self.autoencoder_optimizer = optim.SGD
        else:
            assert False

        total_params = sum(p.numel() for p in itertools.chain(itertools.chain(self.get_generator().parameters(),
                                                                              self.prior.parameters())))
        if self.conditioner is not None:
            total_params += sum([p.numel() for p in self.conditioner.parameters()])
        if self.ae_conditioned:
            autoencoder_classifier = SingleIncrementalTaskSolver(emb_dim)
            autoencoder_classifier.add_task(classes)
            total_params += sum([p.numel() for p in autoencoder_classifier.trainable_parameters()])

        if logger is not None:
            logger.info('Offline NF parameters:')
            # logger.info(F'\tLevels: {self.n_levels}')
            # logger.info(F'\tBlocks: {self.blocks}')
            # logger.info(F'\tN hiddens: {self.n_hidden}')
            # logger.info(F'\tHidden layers dim: {self.hidden_size}')
            logger.info(F'\tAE train epochs: {self.autoencoder_epochs}')
            logger.info(F'\tNF train epochs: {self.generator_epochs}')
            # logger.info(F'\tReconstruction loss: {rec_loss}')
            # logger.info(F'\tTraining batch size: {self.train_batch_size}')
            # logger.info(F'\tRegularization batch size: {self.reg_batch_size}')
            logger.info(F'\tweights: {ws}')
            logger.info(F'\tNumber of parameters: {total_params}')
            logger.info(F'\tEmbedding dim: {self.emb_dim}')

        self.autoencoder_opt = None
        self.hooks = []

        self._d2t = torch.empty(classes, device=device, dtype=torch.long).fill_(-1)

        self._t2d = {}

        self.all_labels = []
        self.old_labels = []
        self.task_labels = []

        self.old_generator = None
        self.old_decoder = None
        self.old_conditioner = None
        self.sampled_images_memory = None
        self.generator = self.get_generator()

    def autoencoder_training(self, container):

        for param in container.encoder.parameters():
            param.requires_grad = True

        _, images, labels = next(iter(container.current_task(self.train_batch_size)))

        if self.ae_conditioned:

            emb = container.encoder(images)
            emb = torch.flatten(emb, 1)

            if self.autoencoder_classifier is None:
                # self.autoencoder_classifier = torch.nn.Sequential(*[
                #     torch.nn.Linear(emb.size(1), emb.size(1) // 2), torch.nn.ReLU(),
                #     torch.nn.Linear(emb.size(1) // 2, self.classes)]).to(self.device)
                self.autoencoder_classifier = SingleIncrementalTaskSolver(emb.size(1)).to(self.device)

            self.autoencoder_classifier.add_task(len(container.current_task.task_labels))
            self.autoencoder_classifier = self.autoencoder_classifier.to(self.device)

            autoencoder_opt = Adam(
                itertools.chain(itertools.chain(self.decoder.parameters(),
                                                self.autoencoder_classifier.trainable_parameters(),
                                                container.encoder.parameters())), lr=0.001)
        else:
            autoencoder_opt = Adam(
                itertools.chain(itertools.chain(self.decoder.parameters(), container.encoder.parameters())), lr=0.001)

        container.current_task.set_labels_type('dataset')
        container.current_task.train()

        hooks = []
        for n, m in itertools.chain(self.decoder.named_modules(), container.encoder.named_modules()):
            if isinstance(m, torch.nn.ReLU):  # or isinstance(m, torch.nn.Conv2d):
                hooks.append(Hook(m))

        # _, x_plot, y_plot = container.current_task.sample(size=self.train_batch_size)
        x_plot = None
        first_plotted = False

        for e in range(self.autoencoder_epochs):
            if self.plot_dir is not None and (e % self.plot_step == 0 or e + 1 == self.autoencoder_epochs):
                if x_plot is not None:
                    if not first_plotted:
                        f = torchvision.utils.make_grid(x_plot.cpu(), scale_each=True).numpy()
                        f = plt.imshow(np.transpose(f, (1, 2, 0)), interpolation='nearest')
                        f.figure.savefig(os.path.join(self.plot_dir, 'images_task{}.png'
                                                      .format(container.current_task.index)))
                        first_plotted = True

                    f = self.plot_rec(x_plot, container)
                    f.savefig(os.path.join(self.plot_dir, 'images_rec_task{}_{}.png'
                                           .format(container.current_task.index,
                                                   e if not e + 1 == self.autoencoder_epochs else 'final')))

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                self.decoder.train()
                container.encoder.train()

                if container.current_task.index > 0 and self.reg_type != 'alternation':
                    old_images, old_labels, old_embeddings = self.get_sampled_images(images.size(0))

                    emb = container.encoder(old_images)
                    dis_reg = torch.sub(1.0, torch.nn.functional.cosine_similarity(torch.flatten(emb, 1),
                                                                                   torch.flatten(old_embeddings, 1),
                                                                                   dim=-1))
                    dis_reg = dis_reg.mean() * self.er

                    images, labels = self.combine_batches(images, old_images, labels, old_labels)
                else:
                    dis_reg = 0

                if i == 0 and e == 0:
                    x_plot = images

                emb = container.encoder(images)

                x_rec = self.decoder(emb)

                assert not torch.isnan(images).any(), 'Sampled Images NaN'
                assert not torch.isnan(emb).any(), 'Emb NaN'
                assert not torch.isnan(x_rec).any(), 'X_rec NaN'

                if self.ae_conditioned:
                    pred = self.autoencoder_classifier(torch.flatten(emb, 1))

                    cross_entropy = torch.nn.functional.cross_entropy(pred,
                                                                      labels.long(), reduction='none')
                    cross_entropy = cross_entropy.mean()
                    cross_entropy *= self.cew
                else:
                    cross_entropy = 0

                rec_loss = self.rec_loss(x_rec, images, reduction='mean')

                l1_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                for h in hooks:
                    l1_loss += torch.abs(h.output).mean()

                loss = rec_loss + cross_entropy + dis_reg + l1_loss * self.l1

                if container.current_task.index > 0 and self.reg_type == 'alternation':
                    images, labels, old_embeddings = self.get_sampled_images(emb.size(0))

                    emb = container.encoder(images)
                    x_rec = self.decoder(emb)

                    if self.ae_conditioned:
                        pred = self.autoencoder_classifier(torch.flatten(emb, 1))

                        cross_entropy = torch.nn.functional.cross_entropy(pred,
                                                                          labels.long(), reduction='none')
                        cross_entropy = cross_entropy.mean()
                        cross_entropy *= self.cew
                    else:
                        cross_entropy = 0

                    dis_reg = torch.sub(1.0, torch.nn.functional.cosine_similarity(torch.flatten(emb, 1),
                                                                                   torch.flatten(old_embeddings, 1),
                                                                                   dim=-1))

                    dis_reg = dis_reg.mean()

                    rec_loss = self.rec_loss(x_rec, images)

                    old_loss = rec_loss + cross_entropy + dis_reg * self.er

                    loss = (1 - self.old_w) * loss + self.old_w * old_loss

                autoencoder_opt.zero_grad()
                loss.backward()
                autoencoder_opt.step()

        for h in hooks:
            h.close()

        for param in container.encoder.parameters():
            param.requires_grad = False

    def nf_training(self, container: Container):

        self.generator = self.get_generator()
        container.encoder.eval()

        parameters = list(self.generator.parameters())
        if self.conditioner is not None:
            parameters.extend(list(self.conditioner.parameters()))

        if self.ae_conditioned and self.conditioner_fine_tuning:
            conditioner_fine_tuning_opt = torch.optim.Adam(list(self.autoencoder_classifier.trainable_parameters()),
                                                           lr=1e-4)

        inn_optimizer = torch.optim.Adam(parameters, lr=self.nf_lr,
                                         weight_decay=self.nf_weight_decay)

        container.current_task.set_labels_type('dataset')

        if False:
            with torch.no_grad():
                indexes = []
                losses = []
                self.decoder.train()
                container.encoder.train()

                for _, (i, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                    emb = container.encoder(images)

                    x_rec = self.decoder(emb)

                    pred = self.autoencoder_classifier(torch.flatten(emb, 1))
                    cross_entropy = torch.nn.functional.cross_entropy(pred,
                                                                      labels.long(), reduction='none')
                    cross_entropy *= self.cew
                    rec_loss = self.rec_loss(x_rec, images, reduction='none').view(cross_entropy.size(0), -1).sum(1)

                    loss = cross_entropy + rec_loss

                    losses.extend(loss)
                    indexes.extend(i)

                # sort desc
                # prendo indici ed estraggo gli indici veri
                # faccio un sampling del sottoinsieme

                losses = torch.stack(losses).tolist()

                values = zip(losses, indexes)
                values = sorted(values, key=lambda x: x[0], reverse=True)
                losses, indexes = zip(*values)
                # indexes = indexes[self.subset_size]

                indexes = indexes[5000:]
                # indexes = torch.tensor(indexes, device=self.device)

                data_to_iter = container.current_task(self.train_batch_size,
                                                      sampler=torch.utils.data.SubsetRandomSampler(indexes))
        else:
            data_to_iter = container.current_task(self.train_batch_size)

        best_model_dict = (None, None)
        best_loss = np.inf

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer, mode='min', factor=0.8,
                                                               patience=self.tolerance // 3 if self.tolerance_training
                                                               else self.generator_epochs // 10,
                                                               verbose=True,
                                                               threshold=0.01, threshold_mode='rel', cooldown=0,
                                                               min_lr=0.00001,
                                                               eps=1e-08)

        u_to_sample = self.prior.sample(100).to(self.device)

        if self.tolerance_training:
            it = itertools.count(start=0, step=1)
            assert self.tolerance > 0
            ctol = self.tolerance
        else:
            it = range(self.generator_epochs)

        for e in it:
            if self.plot_dir is not None and (e % self.plot_step == 0 or e + 1 == self.generator_epochs):
                f = self.plot(u=u_to_sample)
                f.savefig(os.path.join(self.plot_dir, 'sampled_task{}_{}.png'
                                       .format(container.current_task.index,
                                               e if not e + 1 == self.generator_epochs or self.tolerance_training
                                               else 'final')))
                self.calculate_score(container)

            cum_loss = 0

            for i, (_, images, labels) in enumerate(data_to_iter):
                with torch.no_grad():
                    emb = container.encoder(images)

                if container.current_task.index > 0 and self.reg_type != 'alternation':
                    old_images, old_labels, old_emb = self.get_sampled_images(emb.size(0))
                    emb, labels = self.combine_batches(emb, old_emb, labels, old_labels)

                self.generator.train()
                self.prior.train()

                if self.conditioner is not None:
                    self.conditioner.train()
                    u, log_det = self.generator(emb, y=self.conditioner(labels.long()))
                else:
                    u, log_det = self.generator(emb)

                if len(log_det.shape) > 1:
                    log_det = log_det.sum(1)

                log_prob = self.prior.log_prob(u)
                log_prob = torch.flatten(log_prob, 1).sum(1)

                loss = -torch.mean(log_prob + log_det)

                if self.ae_conditioned and self.conditioner_fine_tuning:
                    self.autoencoder_classifier.train()
                    pred = self.autoencoder_classifier(torch.flatten(emb.detach(), 1))
                    cross_entropy = torch.nn.functional.cross_entropy(pred, labels.long(), reduction='mean')
                    loss += cross_entropy

                if container.current_task.index > 0 and self.reg_type == 'alternation':
                    with torch.no_grad():

                        images, labels, old_embeddings = self.get_sampled_images(emb.size(0))
                        emb = container.encoder(images)

                    # u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))
                    u, log_det = self.generator(emb.detach())

                    if len(log_det.shape) > 1:
                        log_det = log_det.sum(1)

                    log_prob = self.prior.log_prob(u)
                    log_prob = torch.flatten(log_prob, 1).sum(1)

                    old_loss = -torch.mean(log_prob + log_det)

                    loss = (1 - self.old_w) * loss + self.old_w * old_loss

                cum_loss += loss.item()

                inn_optimizer.zero_grad()
                if self.ae_conditioned and self.conditioner_fine_tuning:
                    conditioner_fine_tuning_opt.zero_grad()
                loss.backward()
                if self.ae_conditioned and self.conditioner_fine_tuning:
                    conditioner_fine_tuning_opt.step()
                inn_optimizer.step()

            cum_loss /= (i + 1)

            if cum_loss < best_loss:
                best_loss = cum_loss
                best_model_dict = (self.generator.state_dict(),
                                   self.conditioner.state_dict() if self.conditioner is not None else None)
                ctol = self.tolerance
            else:
                ctol -= 1
                if ctol <= 0 and self.tolerance_training:
                    break

            scheduler.step(cum_loss)

        self.calculate_score(container)

        self.generator.load_state_dict(best_model_dict[0])
        if self.conditioner is not None:
            self.conditioner.load_state_dict(best_model_dict[1])

        if self.plot_dir is not None:
            f = self.plot(u=u_to_sample)
            f.savefig(os.path.join(self.plot_dir, 'best_model_task{}.png'.format(container.current_task.index)))

        container.current_task.set_labels_type('task')

    def calculate_score(self, container):
        true_labels = []
        predicted_labels = []

        for j, x, y in container.current_task:
            true_labels.extend(y.tolist())
            emb = container.encoder(x)
            a = self.autoencoder_classifier(emb)
            predicted_labels.extend(a.max(dim=1)[1].tolist())

        a, b = np.asarray(true_labels), np.asarray(predicted_labels)
        print((a == b).sum() / len(a))

    def combine_batches(self, xa, xb, ya, yb, zero_prob: float = None):
        if self.reg_type == 'concatenation':
            return torch.cat((xa, xb), 0), torch.cat((ya, yb), 0)
        else:
            assert xa.size() == xb.size(), '{} <> {}'.format(xa.size(), xb.size())

            if zero_prob is None:
                zero_prob = 1 - len(self.old_labels) / (len(self.all_labels))

            batch_size = xa.size(0)
            mask = torch.distributions.binomial.Binomial(probs=zero_prob).sample((batch_size,)).to(self.device)

            x_mask = mask.clone()
            for i in range(len(xa.shape[1:])):
                x_mask.unsqueeze_(-1)

            x = xa * x_mask + xb * (1 - x_mask)
            y = ya * mask + yb.long() * (1 - mask)
            y = y.long()

            return x, y

    def generate_conditioned_batch(self, size, reconstructor, generative_model, labels, conditioner, prior, u=None):
        reconstructor.eval()

        generative_model.eval()
        conditioner.eval()
        prior.eval()

        probs = torch.zeros(max(labels) + 1, device=self.device)
        for i in labels:
            probs[i] = 1

        m = Categorical(probs)
        y = m.sample(torch.Size([size]))
        y_cond = conditioner(y)
        y = y.long()
        if u is None:
            u = prior.sample(size)
        # else:
        #     assert u.size(0) == size
        embs, _ = generative_model.backward(u, y=y_cond)

        z = reconstructor(embs)

        return z, y, embs

    def generate_predicted_batch(self, size, reconstructor, generative_model, predicter, prior, u=None):
        reconstructor.eval()
        predicter.eval()

        generative_model.eval()
        prior.eval()

        if u is None:
            u = prior.sample(size)
        # else:
        #     assert u.size(0) == size
        embs, _ = generative_model.backward(u, y=None)

        pred = predicter(embs)
        y = pred.max(dim=1)[1]
        y = y.long()

        z = reconstructor(embs)

        return z, y, embs

    @torch.no_grad()
    def get_sampled_images(self, size: int):
        if self.fixed_replay:
            if self.sampled_images_memory is None:
                if self.conditioner is None:
                    images, labels, old_embeddings = self.generate_predicted_batch(self.fixed_replay, self.decoder,
                                                                                   self.generator,
                                                                                   self.autoencoder_classifier,
                                                                                   self.prior)
                else:
                    images, labels, old_embeddings = self.generate_conditioned_batch(self.fixed_replay, self.decoder,
                                                                                     self.generator,
                                                                                     self.old_labels, self.conditioner,
                                                                                     self.prior)

                self.sampled_images_memory = TensorDataset(images, labels, old_embeddings)

            indices = torch.randperm(len(self.sampled_images_memory))[:size]

            images, labels, old_embeddings = self.sampled_images_memory[indices]

            return images, labels, old_embeddings
        else:
            if self.conditioner is None:
                images, labels, old_embeddings = self.generate_predicted_batch(size, self.old_decoder,
                                                                               self.old_generator,
                                                                               self.autoencoder_classifier, self.prior)
            else:
                images, labels, old_embeddings = self.generate_conditioned_batch(size, self.old_decoder,
                                                                                 self.old_generator,
                                                                                 self.old_labels, self.old_conditioner,
                                                                                 self.prior)

            return images, labels, old_embeddings

    def on_task_starts(self, container: Container, *args, **kwargs):

        self.sampled_images_memory = None
        self.all_labels.extend(container.current_task.task_labels)

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

        plt.close('all')
        self.nf_training(container)

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)
        self.sampled_images_memory = None

        if not self.fixed_replay:
            self.old_generator = self.get_generator().to(self.device)
            self.old_generator.load_state_dict(self.generator.state_dict())
            self.old_conditioner = deepcopy(self.conditioner)
            self.old_generator.eval()

            self.old_decoder = deepcopy(self.decoder)
            self.old_decoder.load_state_dict(self.decoder.state_dict())
            self.old_decoder.eval()

    def plot(self, labels=None, n=100, u=None):
        if labels is None:
            labels = []
            for tl in self.task_labels:
                labels.extend(tl)

        labels.sort()
        # sample_n = 10

        with torch.no_grad():
            self.generator.eval()
            self.decoder.eval()

            if self.conditioner is not None:
                x, _, _ = self.generate_conditioned_batch(n, self.decoder, self.generator, labels, self.conditioner,
                                                          self.prior, u)
            else:
                x, _, _ = self.generate_predicted_batch(n, self.decoder, self.generator, self.autoencoder_classifier,
                                                        self.prior, u)

            f = torchvision.utils.make_grid(x.cpu(), scale_each=True, ).numpy()
            f = plt.imshow(np.transpose(f, (1, 2, 0)), interpolation='nearest')

            return f.figure

    def plot_rec(self, x, container):

        with torch.no_grad():
            emb = container.encoder(x)
            x_rec = self.decoder(emb)

        f = torchvision.utils.make_grid(x_rec.cpu(), scale_each=True, ).numpy()
        f = plt.imshow(np.transpose(f, (1, 2, 0)), interpolation='nearest')

        return f.figure

    def get_generator(self):

        gen = RNVP(n_levels=self.levels, levels_blocks=self.blocks, input_dim=self.emb_dim,
                   n_hidden=self.n_hidden, conditioning_size=0, hidden_size=self.hidden_size)

        return gen.to(self.device)
