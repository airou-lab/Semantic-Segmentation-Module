"""
SegFormer model wrapper with PyTorch Lightning
"""

import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
# from datasets import load_metric // old evaluate has been deprecated
from evaluate import load as load_metric
import torch
from torch import nn
import numpy as np
from config import *


class SegformerFinetuner(pl.LightningModule):
    def __init__(
        self,
        id2label=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        metrics_interval=100,
        learning_rate=LEARNING_RATE,
        num_classes=NUM_CLASSES,
        model_name=MODEL_NAME,
    ):
        super(SegformerFinetuner, self).__init__()
        self.save_hyperparameters(ignore=['train_dataloader', 'val_dataloader', 'test_dataloader'])
        
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.learning_rate = learning_rate

        # Use provided id2label or default from config
        self.id2label = id2label if id2label is not None else ID2LABEL
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_classes = num_classes

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            return_dict=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
        self.validation_step_outputs = []
        self.test_outputs = []

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return outputs

    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            metrics = {
                'loss': loss,
                "mean_iou": metrics["mean_iou"],
                "mean_accuracy": metrics["mean_accuracy"]
            }

            for k, v in metrics.items():
                self.log(k, v, prog_bar=True)

            return metrics
        else:
            self.log('loss', loss, prog_bar=True)
            return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        self.validation_step_outputs.append({'val_loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]

        metrics = {
            "val_loss": avg_val_loss,
            "val_mean_iou": val_mean_iou,
            "val_mean_accuracy": val_mean_accuracy
        }
        
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        self.validation_step_outputs.clear()
        return metrics

    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        self.test_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_test_loss = torch.stack(self.test_outputs).mean() if self.test_outputs else torch.tensor(0.0)
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {
            "test_loss": avg_test_loss,
            "test_mean_iou": test_mean_iou,
            "test_mean_accuracy": test_mean_accuracy
        }

        for k, v in metrics.items():
            self.log(k, v)

        self.test_outputs.clear()
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            eps=1e-08
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl