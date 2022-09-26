import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping, Tuple

from deepmhci.evaluation import get_auc, get_pcc, CUTOFF, get_aupr


class Model(object):
    """

    """
    def __init__(self, network, model_path, *, binary=False, earlystop_patience=7, scheduler_patience=5, delta=0, **kwargs):
        self.model = self.network = network(**kwargs).cuda()
        self.binary = binary
        self.loss_fn, self.model_path = nn.MSELoss() if not self.binary else nn.BCELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.earlystop_patience, self.delta, self.counter, self.scheduler_patience = earlystop_patience, delta, 0, scheduler_patience  


    def train_step(self, train_x: Tuple[torch.Tensor, torch.Tensor], train_y: torch.Tensor, **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(*(x.cuda() for x in train_x), **kwargs)
        loss = self.loss_fn(scores, train_y.to(torch.float32).cuda())
        loss.backward()
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, data_x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        score = self.model(*(x.cuda() for x in data_x), **kwargs)
        return score.cpu()

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.scheduler_patience) 

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = (),
              epochs_num=20, verbose=True, board_writer: Optional[SummaryWriter] = None, **kwargs):
        self.get_optimizer(**dict(opt_params))
        global_step, epoch_idx, best_score = 0, 0, 0.0
        for epoch_idx in range(epoch_idx, epochs_num):
            train_loss = 0.0
            for train_x, train_y in tqdm(train_loader, desc=F'Epoch {epoch_idx}', ncols=80, leave=False, dynamic_ncols=False):
                train_loss += self.train_step(train_x, train_y) * len(train_y)
            train_loss /= len(train_loader.dataset)
            best_score = self.valid(valid_loader, best_score, verbose, board_writer, epoch_idx, train_loss)
            self.scheduler.step(best_score)
            if self.counter >= self.earlystop_patience:
                break

    def valid(self, valid_loader, best_score, verbose, board_writer, epoch_idx, train_loss, **kwargs):
        mhc_names, targets = valid_loader.dataset.mhc_names, valid_loader.dataset.targets
        scores = self.predict(valid_loader, valid=True, **kwargs)
        auc, aupr, pcc = get_auc(targets, scores), get_aupr(targets, scores), get_pcc(targets, scores)
        if pcc > best_score + self.delta:
            self.save_model()
            best_score = pcc
            self.counter = 0
        else:
            self.counter += 1
            logger.info(F'EarlyStopping counter: {self.counter} out of {self.earlystop_patience}')
        if verbose:
            logger.info(F'Epoch: {epoch_idx} '
                        F'train loss: {train_loss:.5f} '
                        F'AUC: {auc:.3f} AUPR: {aupr:.3f} PCC: {pcc:.3f}')
        if board_writer is not None:
            board_writer.add_scalar('AUC', auc, epoch_idx)
            board_writer.add_scalar('PCC', pcc, epoch_idx)
            board_writer.add_text('Training Log',
                                  F'Epoch: {epoch_idx} '
                                  F'train loss: {train_loss:.5f} '
                                  F'AUC: {auc:.3f} PCC: {pcc:.3f}', epoch_idx)
        return best_score

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        scores , idxes = [], []
        for data_x, _ in tqdm(data_loader, ncols=80, leave=False, dynamic_ncols=False):
            score = self.predict_step(data_x, **kwargs)
            scores.append(score)
        scores = np.hstack(scores)
        return scores
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

