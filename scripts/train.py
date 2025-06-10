"""
Training script for SegFormer model
"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import SegformerFeatureExtractor
from torch.utils.data import DataLoader
import roboflow
from roboflow import Roboflow

from dataset import SemanticSegmentationDataset
from model import SegformerFinetuner
from config import *


def download_dataset():
    """Download dataset from Roboflow"""
    roboflow.login()
    rf = Roboflow()
    
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("png-mask-semantic")
    return dataset.location


def main(args):
    # Set up dataset path
    if args.dataset_path is None:
        print("No dataset path provided. Downloading from Roboflow...")
        dataset_path = download_dataset()
    else:
        dataset_path = args.dataset_path
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = args.image_size

    # Create datasets
    train_dataset = SemanticSegmentationDataset(
        os.path.join(dataset_path, "train/"), 
        feature_extractor
    )
    val_dataset = SemanticSegmentationDataset(
        os.path.join(dataset_path, "valid/"), 
        feature_extractor
    )
    test_dataset = SemanticSegmentationDataset(
        os.path.join(dataset_path, "test/"), 
        feature_extractor
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # Initialize model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        segformer_finetuner = SegformerFinetuner.load_from_checkpoint(
            args.checkpoint,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            metrics_interval=args.metrics_interval,
        )
    else:
        print("Training from scratch...")
        segformer_finetuner = SegformerFinetuner(
            train_dataset.id2label,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            metrics_interval=args.metrics_interval,
            learning_rate=args.learning_rate,
        )

    # Set up callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='segformer-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Set up logger
    logger = TensorBoardLogger(LOG_DIR, name=args.experiment_name)

    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=args.epochs,
        val_check_interval=len(train_dataloader),
    )

    # Train the model
    trainer.fit(segformer_finetuner)

    # Test the model
    if args.test_after_training:
        print("Running test evaluation...")
        trainer.test(segformer_finetuner)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.experiment_name}_final.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Final checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer on bird camera trap data")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset. If not provided, downloads from Roboflow")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of data loader workers")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Size to resize images")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early stopping patience")
    parser.add_argument("--metrics_interval", type=int, default=METRICS_INTERVAL,
                        help="How often to compute metrics during training")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, default="bird_segmentation",
                        help="Name for this experiment")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use (0 for CPU)")
    
    # Other arguments
    parser.add_argument("--test_after_training", action="store_true",
                        help="Run test evaluation after training")
    
    args = parser.parse_args()
    main(args)