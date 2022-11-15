import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color, dict2str

from utils import my_get_model, my_get_trainer


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    print(model)
    try:
        model = get_model(model)
    except:
        model = my_get_model(model)

    # configurations initialization
    config = Config(model=model, dataset=dataset,
                    config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = model(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = my_get_trainer(
        config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    valid_result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=config['show_progress'])
    valid_result_output = set_color(
        'valid result', 'blue') + ': \n' + dict2str(valid_result)
    logger.info(valid_result_output)

    best_valid_score, best_valid_result = trainer.fit(train_data,
                                                      valid_data,
                                                      saved=saved,
                                                      show_progress=config['show_progress'])

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    print_result(test_result, logger, k=4)
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def print_result(test_result, logger, k=4):
    count = 0
    info = '\ntest result:'
    for i in test_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % k
        info += "{:15}:{:<10}    ".format(i, test_result[i])
    logger.info(info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str,
                        default='TCPSRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str,
                        default='yelp', help='name of datasets')
    parser.add_argument('--config_files', type=str,
                        default='./config/data.yaml ./config/model.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(
        ' ') if args.config_files else None
    print(config_file_list)
    run_recbole(model=args.model, dataset=args.dataset,
                config_file_list=config_file_list)
