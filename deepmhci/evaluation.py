import math
import csv
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from logzero import logger

CUTOFF = 1.0 - math.log(500, 50000)


def get_auc(targets, scores):
    return roc_auc_score(targets >= CUTOFF, scores)


def get_aupr(targets, scores):
    return average_precision_score(targets >= CUTOFF, scores)


def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]


def get_srcc(targets, scores):
    return spearmanr(targets, scores)[0]


def get_frank(targets, scores):
    return ((scores - scores[targets.argmax()]) > 0).sum() / len(scores)


def get_group_metrics(mhc_names, targets, scores, function='allele', reduce=True):
    mhc_names, targets, scores = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores)
    mhc_groups, aucs, auprs, pccs, srccs = [], [], [], [], []
    if function == 'allele':
        for mhc_name_ in sorted(set(mhc_names)):
            t_, s_ = targets[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
            if len(t_) > 20 and len(t_[t_ >= CUTOFF]) >= 3 and len(t_[t_ >= CUTOFF])/len(t_) <= 0.99:
                mhc_groups.append(mhc_name_)
                aucs.append(get_auc(t_, s_))
                auprs.append(get_aupr(t_, s_))
                pccs.append(get_pcc(t_, s_))
                srccs.append(get_srcc(t_, s_))
        return (np.mean(aucs), np.mean(auprs), np.mean(pccs), np.mean(srccs)) if reduce else (mhc_groups, aucs, auprs, pccs, srccs)
    elif function == 'all':
        auc, aupr, pcc, srcc = get_auc(targets, scores), get_aupr(targets, scores), get_pcc(targets, scores), get_srcc(targets, scores)
        return auc, aupr, pcc, srcc


def output_res(mhc_names, targets, scores, output_path: Path, function='allele'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    eval_out_path = output_path.with_suffix('.csv')
    mhc_names, targets, scores, metrics = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores), []
    if function == 'allele':
        with open(eval_out_path, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['allele', 'total', 'positive', 'AUC', 'AUPR', 'PCC', 'SRCC'])
            mhc_groups, aucs, auprs, pccs, srccs = get_group_metrics(mhc_names, targets, scores, function, reduce=False)
            for mhc_name_, auc, aupr, pcc, srcc in zip(mhc_groups, aucs, auprs, pccs, srccs):
                t_ = targets[mhc_names == mhc_name_]
                writer.writerow([mhc_name_, len(t_), len(t_[t_ >= CUTOFF]), auc, aupr, pcc, srcc])
                metrics.append((auc, aupr, pcc, srcc))
            metrics = np.mean(metrics, axis=0)
            writer.writerow(['', '', '', metrics[0], metrics[1], metrics[2], metrics[3]])
        logger.info(F'eval_func: {function} AUC: {metrics[0]:5f} AUPR: {metrics[1]:5f} PCC: {metrics[2]:5f} SRCC: {metrics[3]:5f}')
    elif function == 'all':
        with open(eval_out_path, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['AUC', 'AUPR', 'PCC', 'SRCC'])
            auc, aupr, pcc, srcc = get_group_metrics(mhc_names, targets, scores, function)
            writer.writerow([auc, aupr, pcc, srcc])
        logger.info(F'eval_func: {function} AUC: {auc:5f} AUPR: {aupr:5f} PCC: {pcc:5f} SRCC: {srcc:5f}')

def output_epitope_res(mhc_names, targets, scores, output_path: Path):
    mhc_names, targets, scores = (np.asarray(mhc_names), np.asarray(targets, dtype=object),
                                  np.asarray(scores, dtype=object))
    metrics = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    eval_out_path = output_path.with_suffix('.csv')
    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['allele', 'total', 'Frank', 'AUC', 'AUPR'])
        for mhc_name_ in sorted(set(mhc_names)):
            t_, s_ = targets[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
            frank, auc, aupr = np.mean([(get_frank(tx_, sx_), get_auc(tx_, sx_), get_aupr(tx_, sx_)) for tx_, sx_ in zip(t_, s_)], axis=0)
            writer.writerow([mhc_name_, len(t_), frank, auc, aupr])
            metrics.append((frank, auc, aupr))
        metrics = np.mean(metrics, axis=0)
        writer.writerow(['', '', metrics[0], metrics[1], metrics[2]])
    logger.info(F'Frank: {metrics[0]:5f} AUC: {metrics[1]:5f} AUPR: {metrics[2]:5f}')

def output_results_by_len(scores_list, test_truth, test_mhc_name, peptide_len, res_path, test_file_name, eval_func='allele'):
    test_truth, test_mhc_name, peptide_len = np.asarray(test_truth), np.asarray(test_mhc_name), np.asarray(peptide_len)
    for targetlen in [8, 9, 10, 11, 12]:
        if targetlen < 12:
            pick_id = peptide_len == targetlen
        else:
            pick_id = peptide_len >= targetlen
        avg_scores_list = np.mean(scores_list, axis=0)
        try:
            logger.info(F'{targetlen}-mer metric by {eval_func}')
            output_res(test_mhc_name[pick_id], test_truth[pick_id], avg_scores_list[pick_id],
                    res_path.with_name(F'{res_path.stem}-{test_file_name}-(8-12)-{targetlen}length'), eval_func)  
        except Exception:
            pass
