import argparse
import itertools
import logging
import numpy as np
import yaml
import sys
import os.path
import torch
from torch import optim
import pickle
from copy import deepcopy

from base import get_dataset, classification_score_on_task, get_predictions, get_model
from continual_ai.cl_settings import MultiHeadTaskSolver, SingleIncrementalTaskSolver, MultiTask, SingleIncrementalTask
from continual_ai.cl_strategies.multi_task import ElasticWeightConsolidation, GradientEpisodicMemory, \
    EmbeddingRegularization, LearningWithoutForgetting, PRER
from continual_ai.cl_strategies import NaiveMethod, Container
# from continual_ai.cl_strategies.prer.PRER_custom_nf import PRER

from continual_ai.eval import Accuracy, BackwardTransfer, Evaluator, TotalAccuracy, F1, ExperimentsContainer, TimeMetric
from continual_ai.utils import ExperimentConfig


def my_custom_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)

    logger.setLevel(level)

    log_format = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(logger_name, mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


parser = argparse.ArgumentParser(description='Main.')
parser.add_argument('path', action='store', help='The path of the experiment file to load.')
parser.add_argument('--cuda', '--gpu', action='store', default=-1, help='The gpu to use.', type=int)

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

experiment_file = args.path

experiment_config = yaml.load(open(experiment_file), Loader=yaml.FullLoader)
config = ExperimentConfig(experiment_file)

########################################################################

base_experiment_path = config.train_config['save_path']

if not os.path.exists(base_experiment_path):
    os.makedirs(base_experiment_path)

    with open(os.path.join(base_experiment_path, 'config.yml'), 'w') as outfile:
        yaml.dump(experiment_config, outfile, default_flow_style=False)

print(F'Config file loaded: {experiment_file}.')
print(F'with parameters:\n {config.__str__()}.')

final_train = ExperimentsContainer()
final_test = ExperimentsContainer()

for seed in range(config.train_config['experiments']):

    # if seed > 0:
    #     continue

    torch.manual_seed(seed)
    np.random.seed(seed)
    rs = np.random.RandomState(seed)

    experiment_path = os.path.join(base_experiment_path, F'exp_{str(seed)}')
    print(experiment_path)

    if config.train_config['load'] and os.path.exists(os.path.join(experiment_path, 'final_results.pkl')):

        with open(os.path.join(experiment_path, 'final_results.pkl'), 'rb') as file:
            results = pickle.load(file)

        final_train.add_evaluator(results['train'])
        final_test.add_evaluator(results['test'])

        continue

    config.train_config['save'] = False

    plot_path = os.path.join(experiment_path, 'plots')
    models_path = os.path.join(experiment_path, 'models')
    results_path = os.path.join(experiment_path, 'results')

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(models_path)
        os.makedirs(plot_path)
        os.makedirs(results_path)

    logger = my_custom_logger(os.path.join(experiment_path, 'experiment.log'))
    logger.info(F'Started experiment #{seed} with seed {seed}.')

    logger.info(F'Save path: {experiment_path}.')

    device = 'cpu'

    if torch.cuda.is_available() and args.cuda >= 0 and args.cuda != 'cpu':
        torch.cuda.set_device(args.cuda)
        device = 'cuda'

    print('Device {} will be used'.format(device))

    dataset_loader, preprocessing, image_channels, image_shape, classes = \
        get_dataset(config.cl_config['dataset'], device)

    logger.info(F'Dataset {config.cl_config["dataset"]} loaded.')

    labels_per_task = config.cl_config['label_per_task']
    shuffle_labels = config.cl_config['shuffle_labels']
    cl_problem = config.cl_config['cl_problem']
    mn = config.cl_technique_config['name']

    encoder, decoder, emb_dim = get_model(config.cl_config['dataset'])

    logger.info(F'Model created.')

    logger.info(F'Continual Learning Strategy: {config.cl_technique_config["name"]}.')

    if mn == 'prer':
        config.train_config['save'] = True
        cl_method = PRER(decoder=decoder, emb_dim=emb_dim, config=config, classes=classes, device=device,
                         plot_dir=plot_path, random_state=rs, logger=logger)
    elif mn == 'naive':
        cl_method = NaiveMethod()
    elif mn == 'ewc':
        cl_method = ElasticWeightConsolidation(config=config, random_state=rs, logger=logger)
    elif mn == 'gem':
        cl_method = GradientEpisodicMemory(config=config, random_state=rs, logger=logger)
    elif mn == 'er':
        cl_method = EmbeddingRegularization(config=config, random_state=rs, logger=logger)
    elif mn == 'lwf':
        cl_method = LearningWithoutForgetting(config=config, random_state=rs, logger=logger)
    else:
        assert False

    if cl_problem == 'mt':
        Cl_Dataset = MultiTask(dataset=dataset_loader, random_state=rs,
                               batch_size=config.train_config['batch_size'],
                               labels_per_task=labels_per_task, shuffle_labels=shuffle_labels)

        solver = MultiHeadTaskSolver(input_dim=emb_dim)
        # TODO: implement SIT problem for each method

    elif cl_problem == 'sit':
        Cl_Dataset = SingleIncrementalTask(dataset=dataset_loader, random_state=rs,
                                           batch_size=config.train_config['batch_size'],
                                           labels_per_task=labels_per_task, shuffle_labels=shuffle_labels)
        solver = SingleIncrementalTaskSolver(input_dim=emb_dim, flat_input=True)

    else:
        assert False

    logger.info(F'Continual Learning dataset created.')

    cl_method = cl_method.to(device)

    encoder = encoder.to(device)
    solver = solver.to(device)
    cl_method = cl_method.to(device)

    all_test_results = []
    all_train_results = []

    container = Container()

    container.encoder = encoder
    container.solver = solver

    test_results = Evaluator(classification_metrics=[Accuracy(), F1()],
                             cl_metrics=[BackwardTransfer(), TotalAccuracy()], other_metrics=TimeMetric())
    train_results = Evaluator(classification_metrics=[Accuracy(), F1()],
                              cl_metrics=[BackwardTransfer(), TotalAccuracy()], other_metrics=TimeMetric())

    for task in Cl_Dataset:

        train_results.on_task_starts()

        task_n = task.index

        models_task_path = os.path.join(models_path, F'task_{task_n}.ptc')
        test_results_path = os.path.join(results_path, F'train_{task_n}.pkl')
        train_results_path = os.path.join(results_path, F'test_{task_n}.pkl')

        logger.info(F'Training on task #{task_n} (# training samples {len(task)}) '
                    F'for {config.train_config["epochs"]} epochs')

        logger.info(F'Task labels {task.task_labels}, original labels: {task.dataset_labels}')

        container.current_task = task

        solver.add_task(len(task.task_labels))
        solver = solver.to(device)

        solver.zero_grad()
        encoder.zero_grad()

        lr = config.train_config['lr']
        opt = config.train_config['optimizer']

        if opt == 'adam':
            optimizer = optim.Adam(itertools.chain(solver.trainable_parameters(task_n), encoder.parameters()),
                                   lr=lr)
        elif opt == 'sgd':
            optimizer = optim.SGD(itertools.chain(solver.trainable_parameters(task_n), encoder.parameters()),
                                  lr=lr)
        else:
            assert False

        container.optimizer = optimizer

        cl_method.on_task_starts(container)

        task.set_labels_type('task')

        bets_res = 0
        best_model = (None, None)

        for e in range(config.train_config['epochs']):

            container.current_epoch = e

            cl_method.on_epoch_starts(container)

            if e == 0:
                task.test()
                y_true, y_pred = get_predictions(encoder, solver, task)
                test_results.evaluate(y_true, y_pred, current_task=task.index, evaluated_task=task.index)

                task.train()
                y_true, y_pred = get_predictions(encoder, solver, task)
                train_results.evaluate(y_true, y_pred, current_task=task.index, evaluated_task=task.index)

            task.train()
            task.set_labels_type('task')

            for batch_idx, (indexes, x, y) in enumerate(task):
                encoder.train()
                solver.train()

                container.current_batch = (indexes, x, y)

                cl_method.on_batch_starts(container)

                emb = encoder(x)
                pred = solver(emb, task=task.index)

                container.others_parameters['embeddings'] = emb
                container.others_parameters['predictions'] = pred

                class_ce = torch.nn.functional.cross_entropy(pred, y)
                loss = class_ce

                container.current_loss = loss

                cl_method.before_gradient_calculation(container)

                optimizer.zero_grad()

                loss.backward()

                cl_method.after_back_propagation(container)

                optimizer.step()

                cl_method.after_optimization_step(container)

            cl_method.on_epoch_ends(container)

            for t in [Cl_Dataset[t] for t in range(task.index + 1)]:
                t.set_labels_type('task')

                t.test()
                y_true, y_pred = get_predictions(encoder, solver, t)
                test_results.evaluate(y_true, y_pred, current_task=task.index, evaluated_task=t.index)

                t.train()
                y_true, y_pred = get_predictions(encoder, solver, t)
                train_results.evaluate(y_true, y_pred, current_task=task.index, evaluated_task=t.index)

            if test_results.cl_results()['Accuracy']['TotalAccuracy'] > bets_res:
                best_model = (solver.state_dict(), decoder.state_dict())
                bets_res = test_results.cl_results()['Accuracy']['TotalAccuracy']

        solver.load_state_dict(best_model[0])
        decoder.load_state_dict(best_model[1])

        cl_method.on_task_ends(container)

        if config.train_config['save']:
            with open(test_results_path, 'wb') as file:
                pickle.dump(test_results, file, protocol=pickle.HIGHEST_PROTOCOL)

            with open(train_results_path, 'wb') as file:
                pickle.dump(train_results, file, protocol=pickle.HIGHEST_PROTOCOL)

            all_test_results.append(test_results)
            all_train_results.append(train_results)

            with open(os.path.join(models_path, F'task{task_n}_cl_strategy.cl'), 'wb') as file:
                pickle.dump(cl_method, file, protocol=pickle.HIGHEST_PROTOCOL)

            state = {'encoder': encoder, 'solver': solver, 'optimizer': optimizer}

            torch.save(state, os.path.join(models_path, F'task{task_n}_state.pth'))

        train_results.on_task_ends()

        logger.info(F'Training on task #{task_n} over.\n')

        logger.info(F'Test split results:')
        print(test_results.cl_results())

        for k in test_results.classification_metrics:
            logger.info(F'{k}: \n{test_results.cl_results()[k]}\n{test_results.task_matrix[k]}')

        logger.info(F'Train split results:')
        for k in train_results.classification_metrics:
            logger.info(F'{k}: \n{train_results.cl_results()[k]}\n{train_results.task_matrix[k]}')

        otm = train_results.others_metrics_results().items()
        if len(otm) > 0:
            logger.info(F'Other metrics train:')
            for k, v in otm:
                logger.info(F'{k}: {v}')

        otm = test_results.others_metrics_results().items()
        if len(otm) > 0:
            logger.info(F'Other metrics test:')
            for k, v in otm:
                logger.info(F'{k}: {v}')

        logger.info(F'\n')

        final_train.add_evaluator(train_results)
        final_test.add_evaluator(test_results)

    # if config.train_config['save']:
    with open(os.path.join(experiment_path, 'final_results.pkl'), 'wb') as file:
        pickle.dump({'train': train_results, 'test': test_results}, file,
                    protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Training process complete.')

print('Final score')

logger = my_custom_logger(os.path.join(base_experiment_path, 'final_score.log'))
logger.info('Train score:')

# for k in train_results.classification_metrics:
#     logger.info(F'{k}: \n{train_results.cl_results()[k]}\n{train_results.task_matrix[k]}')

for k, v in final_train.cl_metrics().items():
    logger.info('{}: {}'.format(k, v))
logger.info('\t{}'.format(final_train.others_metrics()))

logger.info('Test score:')
for k, v in final_test.cl_metrics().items():
    logger.info('{}: {}'.format(k, v))
logger.info('\t{}'.format(final_test.others_metrics()))
