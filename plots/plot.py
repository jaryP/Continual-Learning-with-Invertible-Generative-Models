import os
from collections import defaultdict

import numpy as np
import pickle

import sys
import yaml
import os.path as path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns

from continual_ai.eval import ExperimentsContainer


def plot_scores(d):
    scores_figs = []
    for name, ds in d.items():
        test_res = ds['results']['test']

        color = ds.get('color', None)
        linestyle = ds.get('linestyle', None)
        legend = name

        scores = test_res.task_scores()

        for metric, res in scores.items():
            for task, scores in res.items():
                _means = scores['mean'][1:]
                _stds = scores['std'][1:]

                if task == 0:
                    num_epochs = len(_means)

                start = num_epochs - len(_means)

                f = plt.figure('scores_{}_{}'.format(metric, task))
                ax = f.gca()

                r = range(start, num_epochs)

                _ms = savgol_filter(_means[1:], 11, 4)
                _means[1:] = _ms

                _c = ax.plot(r, _means, c=color, label=legend, alpha=0.5, linestyle=linestyle)
                color = _c[0].get_color()

                _stds[1:] = savgol_filter(_stds[1:], 11, 4)

                ax.fill_between(r, _means - _stds, _means + _stds, color=color, alpha=0.05)

        scores_figs = [s for s in plt.get_figlabels() if s.startswith('scores')]
        down = np.inf
        up = -np.inf

        for i in scores_figs:
            f = plt.figure(i)
            ax = f.gca()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Test accuracy')
            ax.set_xlim(0, num_epochs - 1)
            ax.patch.set_facecolor('white')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.grid(True, color="0.9", linestyle='--', linewidth=1)
            ax.margins(x=0)

            d, u = ax.get_ylim()

            down = min(down, d)
            up = max(up, u)

        down = down - down * 0.05
        for i in scores_figs:
            f = plt.figure(i)
            ax = f.gca()
            ax.set_ylim((down, up))
            ax.legend(loc='lower left')

            # f = plt.figure(0)
            # ax = f.gca()

    return scores_figs


def plot_matrices(d):
    #matrix
    m_figs = []
    for name, ds in d.items():
        test_res = ds['results']['test']

        color = ds.get('color', None)
        linestyle = ds.get('linestyle', None)
        legend = name

        for metric, res in test_res.matrix().items():
            # print(res)
            mask = np.zeros_like(res['mean'])
            mask[np.triu_indices_from(mask, k=1)] = True
            # mask[np.eye(len(mask))] = True
            # print(mask)
            with sns.axes_style("white"):
                plt.figure('matrix_{}_{}'.format(metric, name), figsize=[4, 3])
                sns.heatmap(res['mean'], vmin=0, vmax=1, center=0.5, mask=mask,
                            annot=res['mean'], linewidths=.5, cbar=False, cmap='rocket')

    m_figs = [s for s in plt.get_figlabels() if s.startswith('matrix')]
    # down = np.inf
    # up = -np.inf
    #
    # for i in m_figs:
    #     f = plt.figure(i)
    #     ax = f.gca()
    #     ax.set_xlabel('Epochs')
    #     ax.set_ylabel('Test accuracy')
    #     ax.set_xlim(0, num_epochs - 1)
    #
    #     ax.grid(True, color="0.9", linestyle='--', linewidth=1)
    #     ax.margins(x=0)
    #
    #     d, u = ax.get_ylim()
    #
    #     down = min(down, d)
    #     up = max(up, u)
    #
    # for i in m_figs:
    #         f = plt.figure(i)
    #         ax = f.gca()
    #         ax.set_ylim((down, up))
    #         ax.legend(loc='lower left')
    return m_figs


font = {'font.family': 'serif',
        'axes.labelsize': 11,
        'font.size': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': [4*2, 3],
        'text.usetex': False,
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'figure.autolayout': True}

plt.rcParams.update(font)
sns.set_theme()

plot_file = sys.argv[1]
plot_config = yaml.load(open(plot_file), Loader=yaml.FullLoader)

ts = {}

for d in plot_config['files']:
    ts.update({d: plot_config['files'][d]})

for k, ds in ts.items():
    t_config = yaml.load(open(ds['path']), Loader=yaml.FullLoader)

    base_experiment_path = t_config['train_config']['save_path']

    all_metrics = []

    ec_train = ExperimentsContainer()
    ec_test = ExperimentsContainer()

    for seed in range(t_config['train_config']['experiments']):

        experiment_path = path.join(base_experiment_path, F'exp_{str(seed)}')
        final_res_path = path.join(experiment_path, 'final_results.pkl')

        assert path.exists(final_res_path), \
            F'Final results file does not exists: {final_res_path}'
        print(final_res_path)
        with open(final_res_path, 'rb') as file:
            results = pickle.load(file)

        # for i in 'train', 'test':
        #     results[i].add_cl_metric(FinalAccuracy())
        #     results[i].add_cl_metric(LastBackwardTransfer())
        # results.add_metric()

        ec_train.add_evaluator(results['train'])
        ec_test.add_evaluator(results['test'])

    ds['results'] = {'train': ec_train, 'test': ec_test}

um_epochs = None
tasks = set()

for name, ds in ts.items():
    test_res = ds['results']['test']

    color = ds.get('color', None)
    linestyle = ds.get('linestyle', None)
    legend = name

    scores = test_res.task_scores()
    cl_metrics = test_res.cl_metrics()
    print(name, cl_metrics)
    print('\t', test_res.others_metrics())

    # for metric, res in scores.items():
    #     for task, scores in res.items():
    #         _means = scores['mean'][1:]
    #         _stds = scores['std'][1:]
    #
    #         if task == 0:
    #             num_epochs = len(_means)
    #
    #         start = num_epochs - len(_means)
    #
    #         f = plt.figure('scores_{}_{}'.format(metric, task))
    #         ax = f.gca()
    #
    #         r = range(start, num_epochs)
    #
    #         _ms = savgol_filter(_means[1:], 11, 4)
    #         _means[1:] = _ms
    #
    #         _c = ax.plot(r, _means, c=color, label=legend, alpha=0.5)
    #         color = _c[0].get_color()
    #
    #         _stds[1:] = savgol_filter(_stds[1:], 11, 4)
    #
    #         ax.fill_between(r, _means - _stds, _means + _stds, color=color, alpha=0.05)
    #
    # scores_figs = [s for s in plt.get_figlabels() if s.startswith('scores')]
    # down = np.inf
    # up = -np.inf
    #
    # for i in scores_figs:
    #     f = plt.figure(i)
    #     ax = f.gca()
    #     ax.set_xlabel('Epochs')
    #     ax.set_ylabel('Test accuracy')
    #     ax.set_xlim(0, num_epochs - 1)
    #
    #     ax.grid(True, color="0.9", linestyle='--', linewidth=1)
    #     ax.margins(x=0)
    #
    #     d, u = ax.get_ylim()
    #
    #     down = min(down, d)
    #     up = max(up, u)
    #
    # for i in scores_figs:
    #     f = plt.figure(i)
    #     ax = f.gca()
    #     ax.set_ylim((down, up))
    #     ax.legend(loc='lower left')
    #
    #     # f = plt.figure(0)
    #     # ax = f.gca()

save_path = plot_config['plot_path']

if not os.path.exists(save_path):
    os.makedirs(save_path)

for fn in plot_scores(ts):
    f = plt.figure(fn)
    f.savefig(os.path.join(save_path, '{}.pdf'.format(fn)))

# if save_path is not None:
#     for i in plt.get_fignums():
#         f = plt.figure(i)
#         f.savefig(os.path.join(save_path, F'{metric}_{i}.pdf'))
