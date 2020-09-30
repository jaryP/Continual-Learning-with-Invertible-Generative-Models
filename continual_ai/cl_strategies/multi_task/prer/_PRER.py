from collections import Counter
from copy import deepcopy

import itertools

import numpy as np
import os

import logging

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import optim
from torch.nn import NLLLoss
from torch.optim import Adam, SGD

from continual_ai.cl_strategies import NaiveMethod, Container
from .dcgan import DCGAN
from .maf import MAF
from .rnvp import RNVP, ActNorm, BatchNorm
from .student_teacher import StudentTeacher
from .utils import Conditioner, Prior, modify_batch
from continual_ai.utils import ExperimentConfig
from .vae import VAE


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.fn)

    def fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class InverseNFER(NaiveMethod):
    # TODO: valutare segunti modifche: 1) prior per ogni task con una piccola rete che prende un embedding
    #  e genera mu e sigma, 2) Multi head generative model, 3) Miglioramento imamgini con metodi
    #  adversarial (https://arxiv.org/pdf/1704.02304.pdf), 4) Implementare samples delle Y in base all'embedding
    #  delle classi del task corrente (solo se il conditioner utilizza non one_hot) 5) annettere Prior e Conditioner nel
    #  NF (?)

    def __init__(self, decoder: torch.nn.Module, emb_dim: int, config: ExperimentConfig, classes: int, device,
                 plot_dir: str,
                 logger: logging.Logger = None, **kwargs):
        super().__init__()

        self.config = config
        self.device = device
        self.plot_dir = plot_dir
        self.classes = classes

        ws = {'old_tasks': 0.5, 'er': 1, 'l1': 0.0001}
        ws.update(config.cl_technique_config.get('weights', {}))

        self.autoencoder_epochs = config.cl_technique_config.get('autoencoder_epochs', config.train_config['epochs'])
        self.generator_epochs = config.cl_technique_config.get('generator_epochs', config.train_config['epochs'])
        self.train_batch_size = config.cl_technique_config.get('train_batch_size', config.train_config['batch_size'])

        rec_loss = config.cl_technique_config.get('rec_loss', 'bce')
        assert rec_loss in ['bce', 'mse']

        if rec_loss == 'bce':
            self.rec_loss = torch.nn.BCELoss()
        else:
            self.rec_loss = torch.nn.MSELoss()

        prior_std = config.cl_technique_config.get('prior_std', 1)
        assert prior_std > 0

        self.old_w = ws.get('old_tasks', 0.5)
        self.er = ws.get('er', 1)
        self.l1 = ws.get('l1')

        self.cl_problem = config.cl_config['cl_problem']

        assert 0 < self.old_w < 1

        self.base_decoder = deepcopy(decoder)
        self.decoder = decoder
        self.emb_dim = emb_dim

        self.conditioner = Conditioner(config.cl_technique_config, classes, emb_dim=emb_dim)

        self.model = config.cl_technique_config.get('model_type', 'rnvp').lower()
        self.n_levels = config.cl_technique_config.get('n_levels', 1)
        self.blocks = config.cl_technique_config.get('blocks', 5)
        self.n_hidden = config.cl_technique_config.get('n_hidden', 2)
        self.hidden_size = config.cl_technique_config.get('hidden_size', emb_dim)
        self.autoencoder_lr = config.train_config['lr']
        self.z_dim = config.cl_technique_config.get('z_dim', emb_dim)

        # if self.model == 'rnvp':
        #     self.generator = RNVP(n_levels=self.n_levels, levels_blocks=self.blocks, n_hidden=self.n_hidden,
        #                           input_dim=emb_dim, conditioning_size=self.conditioner.size, hidden_size=self.hidden_size)

        # self.generator = self.get_generator()
        self.generator = None

        # for module in self.generator.modules():
        #     if isinstance(module, torch.nn.Linear):
        #         torch.nn.init.orthogonal_(module.weight)
        #         if hasattr(module, 'bias') and module.bias is not None:
        #             module.bias.data.fill_(0)

        if self.model in ['gan', 'vae']:
            self.prior = Prior(self.z_dim, std=prior_std)
        else:
            self.prior = Prior(emb_dim, std=prior_std)

        print(self.generator)

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

        # total_params = sum(p.numel() for p in itertools.chain(self.generator.parameters(),
        #                                                       self.conditioner.parameters(),
        #                                                       self.prior.parameters()))
        total_params  = 0
        if logger is not None:
            logger.info('Offline NF parameters:')
            logger.info(F'\tLevels: {self.n_levels}')
            logger.info(F'\tBlocks: {self.blocks}')
            logger.info(F'\tAE train epochs: {self.autoencoder_epochs}')
            logger.info(F'\tNF train epochs: {self.generator_epochs}')
            logger.info(F'\tReconstruction loss: {rec_loss}')
            logger.info(F'\tTraining batch size: {self.train_batch_size}')
            # logger.info(F'\tRegularization batch size: {self.reg_batch_size}')
            logger.info(F'\tweights: {ws}')
            logger.info(F'\tNumber of parameters: {total_params}')

        self._d2t = torch.empty(classes, device=device, dtype=torch.long).fill_(-1)

        self._t2d = {}

        self.all_labels = []
        self.old_labels = []
        self.task_labels = []

        self.old_generator = None
        self.old_decoder = None
        self.old_conditioner = None

        self.plot_path = os.path.join(self.config.train_config['save_path'], 'plots')

    def on_task_starts(self, container: Container, *args, **kwargs):

        dl = np.zeros(len(container.current_task.dataset_labels), dtype=int)

        for i, j in container.current_task.t2d.items():
            dl[i] = j

        dl = torch.tensor(dl, device=self.device, dtype=torch.long)

        self._t2d[container.current_task.index] = dl

        self.all_labels.extend(container.current_task.task_labels)
        print(self.all_labels, self.old_labels)

        self.autoencoder_training(container)

        # if self.model in ['rnvp', 'maf']:
        #     self.nf_training(container)
        # elif self.model == 'gan':
        #     self.gan_training(container)
        # elif self.model == 'vae':
        #     self.vae_training(container)
        # elif self.model == 'st':
        #     self.st_training(container)

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
                    emb, _ = self.generator.backward(u, y=self.conditioner(y))

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

    # def plot_embs(self, container):

    def autoencoder_training(self, container):

        for param in container.encoder.parameters():
            param.requires_grad = True

        if self.config.cl_config['dataset'] in ['mnist', 'kmnist']:
            autoencoder_opt = SGD(itertools.chain(self.decoder.parameters(),
                                                  container.encoder.parameters()), lr=0.1, momentum=0.9)
        else:
            autoencoder_opt = Adam(itertools.chain(self.decoder.parameters(),
                                                   container.encoder.parameters()), lr=0.001)

        container.current_task.set_labels_type('dataset')

        container.current_task.train()

        hooks = []
        for n, m in itertools.chain(self.decoder.named_modules(), container.encoder.named_modules()):
            if isinstance(m, torch.nn.ReLU): # or isinstance(m, torch.nn.Conv2d):
                hooks.append(Hook(m))

        _, x_plot, y_plot = container.current_task.sample(size=self.train_batch_size)

        for e in range(self.autoencoder_epochs):
            tot_sparsity = []

            if e % 5 == 0 and e > 0:
                f = self.plot_rec(x_plot, container)
                f.savefig(os.path.join(self.plot_dir, '{}_rec_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            prior=self.prior,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            return_mask_embeddings=True)

                        if e == 0 and i == 0:
                            x_plot = images

                self.decoder.train()
                container.encoder.train()

                emb = container.encoder(images)

                x_rec = self.decoder(emb)

                dis_reg = torch.tensor(0, dtype=torch.float)

                assert not torch.isnan(images).any(), 'Sampled Images NaN'
                assert not torch.isnan(emb).any(), 'Emb NaN'
                assert not torch.isnan(x_rec).any(), 'X_rec NaN'

                if container.current_task.index > 0:
                    den = mask.sum()
                    if den > 0:
                        mask = 1 - mask
                        dis_reg = 1 - torch.nn.functional.cosine_similarity(emb, old_embeddings, dim=-1, eps=1e-4)
                        dis_reg = dis_reg * mask
                        dis_reg = dis_reg.sum() / den

                    assert not torch.isnan(dis_reg).any(), 'dis_reg Images NaN'

                rec_loss = self.rec_loss(x_rec, images)

                l1_loss = 0
                for h in hooks:
                    l1_loss += torch.abs(h.output).sum(1).mean()

                l1_loss = torch.div(l1_loss, len(hooks))

                loss = rec_loss + dis_reg * self.er + l1_loss * self.l1  # pred_loss * gamma[e]

                tot_sparsity.extend(torch.abs(emb).sum(-1).tolist())

                # bs = images.shape[0]
                #
                # images = images.view((bs, -1))
                # x_rec = x_rec.view((bs, -1))
                #
                # d1 = torch.mm(images, images.t())
                # d2 = torch.mm(x_rec, x_rec.t())
                #
                # # reg_loss = torch.nn.functional.mse_loss(d1, d2)
                # reg_loss = ((d1 - d2)**2).mean(1)
                #
                # print(reg_loss.mean())
                # # reg_loss = torch.nn.functional.pdist(images.view(bs, -1) - x_rec.view(bs, -1))
                # loss += reg_loss.mean() * 0.001

                autoencoder_opt.zero_grad()
                loss.backward()
                autoencoder_opt.step()

            # print(e, np.mean(tot_sparsity))

        for h in hooks:
            h.close()

        f = self.plot_rec(x_plot, container)
        f.savefig(os.path.join(self.plot_dir, 'reconstructed_images_task{}.png'.format(container.current_task.index)))
        plt.close(f)

        for param in container.encoder.parameters():
            param.requires_grad = False

    def nf_training(self, container: Container):

        container.encoder.eval()

        # self.generator = self.get_generator()

        # for m in self.generator.modules():
        #     if isinstance(m, ActNorm) or isinstance(m, BatchNorm):
        #         m.reset_parameters()
        #         m.to(self.device)

        inn_optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters(),
                                                         self.conditioner.parameters()), lr=1e-3, weight_decay=1e-5)

        container.current_task.set_labels_type('dataset')
        # to_plot = []
        to_plot = list(itertools.chain(*self.task_labels, container.current_task.dataset_labels))

        # nll = NLLLoss()

        best_model_dict = None
        best_loss = np.inf

        max_tolerance = 30
        # lr_tolerance = 5
        tol = max_tolerance

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer, mode='min', factor=0.8,
                                                               patience=max_tolerance//2,
                                                               verbose=True,
                                                               threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

        for e in range(self.generator_epochs):
            print(e)
            if e % 5 == 0 and e > 0:
                f = self.plot(to_plot)
                f.savefig(os.path.join(self.plot_dir, '{}_sampled_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            lp = 0

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            prior=self.prior,
                                                                            return_mask_embeddings=True)
                    emb = container.encoder(images)

                self.generator.train()
                self.prior.train()
                self.conditioner.train()

                u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))

                if len(log_det.shape) > 1:
                    log_det = log_det.sum(1)

                log_prob = self.prior.log_prob(u).sum(1)
                loss = -torch.mean(log_prob + log_det)

                lp += loss.item()

                inn_optimizer.zero_grad()
                loss.backward()

                inn_optimizer.step()

            lp /= (i+1)

            if lp < best_loss:
                best_loss = lp
                best_model_dict = self.generator.state_dict()
                tol = max_tolerance
            else:
                tol -= 1
                # print(lp, best_loss, tol)
                if tol == 0:
                    # print('BREAK')
                    break

            scheduler.step(lp)

            # print(lp, log_det.mean().item(), log_prob.mean().item(), u.mean().item(), u.std().item())

        container.current_task.set_labels_type('task')

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)

        f = self.plot()
        f.savefig(os.path.join(self.plot_dir, 'sampled_images_task{}.png'.format(container.current_task.index)))
        plt.close(f)

        self.generator.load_state_dict(best_model_dict)

        old_nf = self.get_generator()

        old_nf.load_state_dict(self.generator.state_dict())

        self.old_generator = old_nf.to(self.device)

        self.old_decoder = deepcopy(self.decoder)

        self.old_conditioner = deepcopy(self.conditioner)

        self.old_generator.eval()
        self.old_decoder.eval()

    def gan_training(self, container: Container):

        container.encoder.eval()

        optimizerD = optim.Adam(self.generator.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.generator.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        criterion = torch.nn.BCELoss()
        container.current_task.set_labels_type('task')

        real_label = 1
        fake_label = 0

        for e in range(self.generator_epochs):
            if e % 5 == 0:
                f = self.plot(container.current_task.dataset_labels)
                f.savefig(os.path.join(self.plot_dir, '{}_sampled_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            for i, (_, images, labels) in enumerate(container.current_task):
                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            prior=self.prior,
                                                                            return_mask_embeddings=True)
                    emb = container.encoder(images)

            b_size = images.shape[0]

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Train with all-real batch
            optimizerD.zero_grad()

            label = torch.full((b_size,), real_label, device=images.device)
            # Forward pass real batch through D
            output = self.generator.D(emb, y=self.conditioner(labels.long())).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with all-fake batch

            noise = self.prior.sample(b_size)
            # Generate fake image batch with G
            fake = self.generator.G(noise, y=self.conditioner(labels.long()))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = self.generator.D(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optimizerG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.generator.D(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()

        # for e in range(self.generator_epochs):
        #     if e % 5 == 0:
        #         f = self.plot(container.current_task.dataset_labels)
        #         f.savefig(os.path.join(self.plot_dir, '{}_sampled_images_task{}.png'
        #                                .format(e, container.current_task.index)))
        #         plt.close(f)
        #
        #     for i, (_, images, labels) in enumerate(container.current_task):
        #         with torch.no_grad():
        #             if container.current_task.index > 0:
        #                 zero_prob = len(self.old_labels) / (len(self.all_labels))
        #                 images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
        #                                                                     self.old_generator,
        #                                                                     self.old_labels,
        #                                                                     conditioner=self.old_conditioner,
        #                                                                     zero_prob=zero_prob,
        #                                                                     prior=self.prior,
        #                                                                     return_mask_embeddings=True)
        #             emb = container.encoder(images)
        #
        #         self.generator.train()
        #         self.prior.train()
        #         self.conditioner.train()
        #
        #         u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))
        #
        #         log_prob = -torch.mean(log_det + self.prior.log_prob(u).sum(1))
        #
        #         inn_optimizer.zero_grad()
        #         log_prob.backward()
        #
        #         inn_optimizer.step()

        container.current_task.set_labels_type('task')

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)

        f = self.plot()
        f.savefig(os.path.join(self.plot_dir, 'sampled_images_task{}.png'.format(container.current_task.index)))
        plt.close(f)

        old_generator = deepcopy(self.generator)

        # old_nf.load_state_dict(self.generator.state_dict())

        self.old_generator = old_generator.to(self.device)

        self.old_decoder = deepcopy(self.decoder)

        self.old_conditioner = deepcopy(self.conditioner)

        self.old_generator.eval()
        self.old_decoder.eval()

    def vae_training(self, container: Container):

        container.encoder.eval()

        # self.generator = self.get_generator()

        inn_optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters(),
                                                         self.conditioner.parameters()), lr=1e-3, weight_decay=1e-5)
        container.current_task.set_labels_type('dataset')

        for e in range(self.generator_epochs):
            print(e)
            if e % 5 == 0 and e > 0:
                f = self.plot(container.current_task.dataset_labels)
                f.savefig(os.path.join(self.plot_dir, '{}_sampled_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                if i == 0:
                    print('prima', Counter(labels.tolist()))

                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            prior=self.prior,
                                                                            return_mask_embeddings=True)
                        if i == 0:
                            print('Dopo', Counter(labels.tolist()))

                    emb = container.encoder(images)

                self.generator.train()
                self.prior.train()
                self.conditioner.train()

                emb_hat, mu, logvar = self.generator(emb, self.conditioner(labels.long()))

                rec = torch.nn.functional.mse_loss(emb_hat, emb)
                kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))

                # log_prob = -torch.mean(log_det + self.prior.log_prob(u).sum(1))
                loss = rec + 0.1 * kdl

                inn_optimizer.zero_grad()
                loss.backward()

                inn_optimizer.step()

        # for param in container.encoder.parameters():
        #     param.requires_grad = False

        container.current_task.set_labels_type('task')

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)

        f = self.plot()
        f.savefig(os.path.join(self.plot_dir, 'sampled_images_task{}.png'.format(container.current_task.index)))
        plt.close(f)

        old_nf = self.get_generator()

        old_nf.load_state_dict(self.generator.state_dict())

        self.old_generator = old_nf.to(self.device)

        self.old_decoder = deepcopy(self.decoder)

        self.old_conditioner = deepcopy(self.conditioner)

        self.old_generator.eval()
        self.old_decoder.eval()

    def st_training(self, container: Container):

        container.encoder.eval()

        inn_optimizer = torch.optim.Adam(itertools.chain(self.generator.teacher.parameters(),
                                                         self.conditioner.parameters()), lr=1e-3, weight_decay=1e-5)
        container.current_task.set_labels_type('dataset')

        # to_plot = list(itertools.chain(*self.task_labels, container.current_task.dataset_labels))

        for e in range(self.generator_epochs):
            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            prior=self.prior,
                                                                            return_mask_embeddings=True)
                    emb = container.encoder(images)

                self.generator.train()
                self.prior.train()
                self.conditioner.train()

                u, log_det = self.generator.teacher(
                    emb.detach(),
                    y=self.conditioner(labels.long()))

                log_prob = -torch.mean(log_det.sum(1) + self.prior.log_prob(u).sum(1))

                inn_optimizer.zero_grad()
                log_prob.backward()

                inn_optimizer.step()

        inn_optimizer = torch.optim.Adam(itertools.chain(self.generator.student.parameters()), lr=1e-3,
                                         weight_decay=1e-5)
        container.current_task.set_labels_type('dataset')

        to_plot = list(itertools.chain(*self.task_labels, container.current_task.dataset_labels))

        # with torch.autograd.set_detect_anomaly(True):
        for e in range(self.generator_epochs):
            if e % 5 == 0 and e > 0:
                f = self.plot(to_plot)
                f.savefig(os.path.join(self.plot_dir, '{}_sampled_images_task{}.png'
                                       .format(e, container.current_task.index)))
                plt.close(f)

            for i, (_, images, labels) in enumerate(container.current_task(self.train_batch_size)):
                with torch.no_grad():
                    if container.current_task.index > 0:
                        zero_prob = len(self.old_labels) / (len(self.all_labels))
                        images, labels, mask, old_embeddings = modify_batch(images, labels, self.old_decoder,
                                                                            self.old_generator,
                                                                            self.old_labels,
                                                                            conditioner=self.old_conditioner,
                                                                            zero_prob=zero_prob,
                                                                            prior=self.prior,
                                                                            return_mask_embeddings=True)
                    # emb = container.encoder(images)

                self.generator.train()
                self.prior.train()
                self.conditioner.train()

                u = self.prior.sample(images.shape[0])

                x, log_det = self.generator.student(u, y=self.conditioner(labels.long()))
                log_prob_student = log_det + self.prior.log_prob(u)
                prob_student = log_prob_student.exp().softmax(-1)

                u_hat, log_det = self.generator.teacher(x, y=self.conditioner(labels.long()))
                log_prob_teacher = log_det + self.prior.log_prob(u_hat)
                prob_teacher = log_prob_teacher.exp().softmax(-1)

                # kl = (prob_student * (prob_student/prob_teacher)).sum(1)
                #
                # kl = kl.mean()
                # h_ps = -log_prob_student.mean()
                #
                # log_prob_student = log_prob_student.exp()
                # log_prob_teacher = log_prob_teacher.exp()
                #
                # h_ps_pt = torch.nn.functional.cross_entropy(log_prob_student, log_prob_teacher)
                #
                # kl = h_ps + h_ps_pt

                # kl = log_prob_student - log_prob_teacher
                # kl = kl.mean(0)
                # kl = (prob_teacher*(a2.log()-a1.log())).sum(1).mean()
                kl = torch.nn.functional.kl_div(prob_student.log(), prob_teacher, reduction='batchmean')
                print(kl)

                if kl == 0:
                    break

                # u, log_det = self.generator(emb.detach(), y=self.conditioner(labels.long()))

                inn_optimizer.zero_grad()
                kl.backward()

                inn_optimizer.step()

        container.current_task.set_labels_type('task')

        self.old_labels.extend(container.current_task.dataset_labels)
        self.task_labels.append(container.current_task.dataset_labels)

        f = self.plot()
        f.savefig(os.path.join(self.plot_dir, 'sampled_images_task{}.png'.format(container.current_task.index)))
        plt.close(f)

        old_nf = self.get_generator()

        old_nf.load_state_dict(self.generator.state_dict())

        self.old_generator = old_nf.to(self.device)

        self.old_decoder = deepcopy(self.decoder)

        self.old_conditioner = deepcopy(self.conditioner)

        self.old_generator.eval()
        self.old_decoder.eval()

    def get_generator(self):
        if self.model == 'nf':
            gen = RNVP(n_levels=self.n_levels, levels_blocks=self.blocks, n_hidden=self.n_hidden,
                       input_dim=self.emb_dim, conditioning_size=self.conditioner.size,
                       hidden_size=self.hidden_size)
        elif self.model == 'gan':
            gen = DCGAN(emb_size=self.emb_dim, hidden_dimension=self.hidden_size, depth=self.n_levels,
                        cond_dim=self.conditioner.size, z_dim=self.z_dim)
        elif self.model == 'maf':
            gen = MAF(n_blocks=self.blocks, input_size=self.emb_dim, conditioning_size=self.conditioner.size,
                      n_hidden=self.n_hidden, hidden_size=self.hidden_size, batch_norm=True)
        elif self.model == 'vae':
            gen = VAE(input_size=self.emb_dim, conditioning_size=self.conditioner.size,
                      n_hidden=self.n_hidden, z_dim=self.z_dim)
        elif self.model == 'st':
            gen = StudentTeacher(n_blocks=self.blocks, input_size=self.emb_dim, conditioning_size=self.conditioner.size,
                                 n_hidden=self.n_hidden, hidden_size=self.hidden_size)
        else:
            assert False

        return gen.to(self.device)
