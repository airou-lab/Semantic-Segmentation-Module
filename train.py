from src.dataset import SemanticSegmentationDataset
from src.model import SegformerFinetuner
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse

# Feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.do_reduce_labels = False
feature_extractor.size = 128

# Daset location (input user)
parser = argparse.ArgumentParser(description="SegFormer Training Script")
parser.add_argument('--dataset_location', type=str, required=True,
                    help='Path to the dataset root directory (should contain train/, valid/, test/)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Optional path to a Lightning checkpoint to resume training')
args = parser.parse_args()
dataset_location = args.dataset_location

# Dataset and dataloaders 
train_dataset = SemanticSegmentationDataset(f"{dataset_location}/train/", feature_extractor)
val_dataset = SemanticSegmentationDataset(f"{dataset_location}/valid/", feature_extractor)
test_dataset = SemanticSegmentationDataset(f"{dataset_location}/test/", feature_extractor)

batch_size = 96
num_workers = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# Model Instantiation
segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    metrics_interval=10,
)

# Callbacks and logger (for plots)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,  #in case validation doesn't improve much in 5 epchos might change to 10 later
    verbose=False,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
logger = TensorBoardLogger("lightning_logs", name="Combined_v1") # Change these names according to trials

# Trainer 
trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=1,
    # precision=16,  # Mixed precision disabled
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=100,
    val_check_interval=len(train_dataloader),
)

# Training 
# Here you have two options: train from scratch or load a pretrained model
if args.checkpoint_path and os.path.exists(args.checkpoint_path):
    print(f"Resuming training from checkpoint: {args.checkpoint_path}")
    trainer.fit(segformer_finetuner, ckpt_path=args.checkpoint_path)
else:
    print("Training model from scratch.")
    trainer.fit(segformer_finetuner)

# Testing
res = trainer.test(segformer_finetuner)


# To manually save a checkpoint (uncomment this)
# trainer.save_checkpoint("bird_project_v1.ckpt")