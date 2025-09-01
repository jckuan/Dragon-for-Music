# coding: utf-8


"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os

def quick_start(model, dataset, config_dict, save_model=True):
    ## Initial setup
    # Merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('█ Server: \t' + platform.node())
    logger.info('█ Data  : \t' + config['data_path'])
    logger.info('█ Dir   : \t' + os.getcwd())

    ## Dataset Preparation
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('█ Dataset   : ' + str(dataset))       # print dataset statistics
    logger.info('█ Training  : ' + str(train_dataset))
    logger.info('█ Validation: ' + str(valid_dataset))
    logger.info('█ Testing   : ' + str(test_dataset))

    ## DataLoader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ## Run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0
    logger.info('\n\n=================================\n\n')

    ## Hyperparameter Setup
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    
    combinators = list(product(*hyper_ls))      # Combinations of hyperparameters
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # Set hyperparameters
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])
        logger.info(f'========={idx+1}/{total_loops}: Parameters:{config["hyper_parameters"]}={hyper_tuple}=======')

        # Initialize model and trainer
        train_data.pretrain_setup()
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer()(config, model)

        # Model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # Results tracking
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    ## Report all results from different hyperparameter combinations
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

def quick_start_fix_params(model, dataset, config_dict, save_model=True):
    ## Initial setup
    # Merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('█ Server: \t' + platform.node())
    logger.info('█ Data  : \t' + config['data_path'])
    logger.info('█ Dir   : \t' + os.getcwd())

    ## Dataset Preparation
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('█ Dataset   : ' + str(dataset))       # print dataset statistics
    logger.info('█ Training  : ' + str(train_dataset))
    logger.info('█ Validation: ' + str(valid_dataset))
    logger.info('█ Testing   : ' + str(test_dataset))

    ## DataLoaders
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))
    
    ## Hyperparameter Setup
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for param in config['hyper_parameters']:        # Update config with the first value for each hyperparameter or None
        config[param] = config[param][0]

    val_metric = config['valid_metric'].lower()
    logger.info(f'█ Parameters: {config["hyper_parameters"]}')

    ## Run model
    # Initialize model and trainer
    init_seed(config['seed'])
    train_data.pretrain_setup()
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer()(config, model)

    # Model training
    best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
        train_data, valid_data=valid_data, test_data=test_data, saved=save_model)

    logger.info(f'Best valid result: {dict2str(best_valid_result)}')
    logger.info(f'Test result: {dict2str(best_test_upon_valid)}')