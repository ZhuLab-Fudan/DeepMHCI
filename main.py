import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
from logzero import logger
import deepmhci

from deepmhci.networks import DeepMHCI
from deepmhci.data_utils import get_data, get_epitopes_data, get_mhc_name_seq, get_raw_data
from deepmhci.datasets import MHCIDataset
from deepmhci.models import Model
from deepmhci.evaluation import output_res, output_epitope_res, output_results_by_len

def train(model, data_cnf, model_cnf, train_data, valid_data=None, collate_fn=None, random_state=1240):
    logger.info(F'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 1000),
                                                  random_state=random_state)
    train_loader = DataLoader(MHCIDataset(train_data, **model_cnf['padding']),
                              batch_size=model_cnf['train']['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(MHCIDataset(valid_data, **model_cnf['padding']),
                              batch_size=model_cnf['valid']['batch_size'], collate_fn=collate_fn)
    model.train(train_loader, valid_loader, **model_cnf['train'])
    logger.info(F'Finish training model {model.model_path}')


def test(model, model_cnf, test_data, collate_fn=None):
    data_loader = DataLoader(MHCIDataset(test_data, **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'], collate_fn=collate_fn)
    return model.predict(data_loader)


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(('train', 'eval', '5cv', 'test-5cv', 'seq2logo', 'epitope')), default=None)
@click.option('--eval_func', type=click.Choice(('allele', 'all')), default='allele')
@click.option('--eval_len', is_flag=True)
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--models-num', default=20)
@click.option('-c', '--continue', 'continue_train', is_flag=True)
@click.option('-a', '--allele', default=None)
def main(data_cnf, model_cnf, mode, continue_train, start_id, models_num, allele, eval_func, eval_len):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    logger.info(F'Model Name: {model_name}')
    model_path = Path(model_cnf['path'])/F'{model_name}'
    res_path = Path(data_cnf['results'])/F'{model_name}'
    model_cnf.setdefault('ensemble', 20)
    binary = data_cnf['binary']

    if mode is None or mode == 'train' or mode == 'eval':
        train_data = get_data(data_cnf['train'], data_cnf['mhc_seq']) if mode is None or mode == 'train' else None
        valid_data = get_data(data_cnf['valid'], data_cnf['mhc_seq']) \
            if (mode is None or mode == 'train') and 'valid' in data_cnf else None
        if mode is None or mode == 'eval':
            test_data = get_data(data_cnf['test'], data_cnf['mhc_seq'])
            test_file_name = Path(data_cnf['test']).stem
            test_mhc_name, test_truth, peptide_len = np.asarray([x[0] for x in test_data]), np.asarray([x[-1] for x in test_data]), np.asarray([len(x[1]) for x in test_data])
        else:
            test_data = test_mhc_name = test_truth = None
        scores_list = []
        for model_id in range(start_id, start_id + models_num):
            model = Model(DeepMHCI, binary=binary,  model_path=model_path.with_name(F'{model_path.stem}-{model_id}'),
                          **model_cnf['model'])
            if train_data is not None:
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data, valid_data)
            if test_data is not None:
                scores = test(model, model_cnf, test_data)
                scores_list.append(scores)
                output_res(test_mhc_name, test_truth, np.mean(scores_list, axis=0), res_path.with_name(F'{res_path.stem}-{test_file_name}'), eval_func)
                if eval_len:
                    output_results_by_len(scores_list, test_truth, test_mhc_name, peptide_len, res_path, test_file_name, eval_func)
            
    elif mode == '5cv':
        data = np.asarray(get_data(data_cnf['train'], data_cnf['mhc_seq']), dtype=object)
        data_mhc_name, data_truth, peptide_len = np.asarray([x[0] for x in data]), np.asarray([x[-1] for x in data]), np.asarray([len(x[1]) for x in data])
        test_file_name = Path(data_cnf['train']).stem
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        with open(data_cnf['eval_id']) as fp:
            eval_id = np.asarray([int(line) for line in fp])
        scores_list = []
        for model_id in range(start_id, start_id + models_num):
            scores_ = np.empty(sum(eval_id), dtype=np.float32)
            for cv_ in range(5):
                train_data, valid_data_tmp, eval_id_tmp = data[cv_id != cv_], data[cv_id == cv_], eval_id[cv_id == cv_]
                valid_data = valid_data_tmp[eval_id_tmp == 1]
                model = Model(DeepMHCI, binary=binary, model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'),
                              **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data, valid_data)
                tmp_cv_id = cv_id[eval_id == 1]
                scores_[tmp_cv_id == cv_] = test(model, model_cnf, valid_data)
            scores_list.append(scores_)
            output_res(data_mhc_name[eval_id == 1], data_truth[eval_id == 1], np.mean(scores_list, axis=0),
                       res_path.with_name(F'{res_path.stem}-5CV'))
            if eval_len:
                output_results_by_len(scores_list, data_truth[eval_id == 1], data_mhc_name[eval_id == 1], peptide_len[eval_id == 1], res_path, test_file_name, eval_func)
            
    elif mode == 'test-5cv':
        cv_file_name = Path(data_cnf['train']).stem
        test_data = get_data(data_cnf['test'], data_cnf['mhc_seq'])
        test_file_name = Path(data_cnf['test']).stem
        test_mhc_name, test_truth, peptide_len = np.asarray([x[0] for x in test_data]), np.asarray([x[-1] for x in test_data]), np.asarray([len(x[1]) for x in test_data])
        scores_list = []
        for model_id in range(start_id, start_id + models_num):
            for cv_ in range(5):
                model = Model(DeepMHCI, binary=binary, model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'), 
                              **model_cnf['model'])
                scores = test(model, model_cnf, test_data)
                scores_list.append(scores)
        test_file_name = F'{cv_file_name}-{test_file_name}'
        output_res(test_mhc_name, test_truth, np.mean(scores_list, axis=0), res_path.with_name(F'{res_path.stem}-{test_file_name}'), eval_func)
        if eval_len:
            output_results_by_len(scores_list, test_truth, test_mhc_name, peptide_len, res_path, test_file_name, eval_func)
            
    elif mode == 'seq2logo':
        mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
        assert allele in mhc_name_seq
        peptide_list, data_list = get_raw_data(data_cnf['seq2logo'], allele, mhc_name_seq[allele],
                                 model_cnf['model']['peptide_pad'])
        scores_list = []
        for model_id in range(start_id, start_id + models_num):
            for cv_ in range(5):
                model = Model(DeepMHCI, binary=binary, model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'),
                            **model_cnf['model'])
                scores = test(model, model_cnf, data_list)
                scores_list.append(scores)
        scores = np.mean(scores_list, axis=0).reshape(len(peptide_list), -1)
        s_ = scores.max(axis=1)
        seq2logo_filename = Path(data_cnf['seq2logo'])
        with open(res_path.with_name(F'{res_path.stem}-{seq2logo_filename.stem}-{allele}.txt'), 'w') as fp:
            for k in (-s_).argsort()[:int(0.01 * len(s_))]:
                print(peptide_list[k], file=fp)
            
    elif mode == 'epitope':
        epitope_path = Path(data_cnf['epitope'])
        data = get_epitopes_data(data_cnf['epitope'], data_cnf['mhc_seq'])
        data_mhc_name, data_targets, data_targets_len = [x[0] for x in data], [np.asarray([y[-1] for y in x[1]]) for x in data], [x[-1] for x in data]
        scores_list = [[] for _ in range(len(data))]
        for model_id in range(start_id, start_id + models_num):
            for cv_ in range(5):
                model = Model(DeepMHCI, binary=binary, model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'),
                            **model_cnf['model'])
                for i, (_, data_list, pep_len) in enumerate(tqdm(data, leave=False, desc=F'Predict: model_id{model_id}-{cv_}')):
                    s_ = test(model, model_cnf, data_list)
                    scores_list[i].append(s_)
                output_epitope_res(data_mhc_name, data_targets, [np.mean(x, axis=0) for x in scores_list],
                                res_path.with_name(F'{res_path.stem}-{epitope_path.stem}-Epitope'))   
        if eval_len:
            data_mhc_name, data_targets, data_targets_len = np.asarray(data_mhc_name), np.asarray(data_targets), np.asarray(data_targets_len)
            saved_scores = np.load(res_path.with_name(F'{res_path.stem}-{epitope_path.stem}-Epitope.npy'), allow_pickle=True)
            for l in [8,9,10,11,12,13]:
                dmn_, dt_ = data_mhc_name[data_targets_len == l], data_targets[data_targets_len == l]
                s_ = saved_scores[data_targets_len == l] 
                output_epitope_res(dmn_, dt_, s_,
                            res_path.with_name(F'{res_path.stem}-{epitope_path.stem}-Epitope-{l}mer'))   
    
if __name__ == '__main__':
    main()
    