#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np

import models
from utils import losses
from utils import Logger
from trainer import Trainer

from src.pascal_data import PascalDataloader
from src.utils import get_instance

class TrainerPascal(Trainer):
    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU[1:].mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Other": {
            "Mean_IoU_upper_body": np.round(IoU[[1,6,2,4]].mean(), 3),
            "Mean_IoU_lower_body": np.round(IoU[[3,5]].mean(), 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
            }
        }

def main():

    config = json.load(open('config_pascal.json'))

    train_loader = PascalDataloader(**config['train_loader']['args'], **config['img_norm'])
    val_loader = PascalDataloader(**config['val_loader']['args'], **config['img_norm'])

    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)

    train_logger = Logger()

    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    trainer = TrainerPascal(
        model=model,
        loss=loss,
        resume=None,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

main()