from lightning.pytorch import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
from torchmetrics import Accuracy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import logging
import torch

log = logging.getLogger(__name__)

class GeneralVisionClassifier(LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_classes: int = 2,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """GeneralVisionClassifier.

        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
            num_classes: Number of classes in the dataset
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_classes = num_classes
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)

    def __build_model(self):
        """Define model layers & loss."""
        # 1. Load pre-trained network:
        backbone = getattr(models, self.backbone)(weights="DEFAULT")
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        in_features = backbone.fc.in_features
        if self.num_classes == 2:  
            _fc_layers = [
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ]
        else:
            _fc_layers = [
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_classes)
            ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = nn.CrossEntropyLoss() if self.num_classes > 2 else F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass.

        Returns logits.
        """
        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        return self.fc(x)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits) if self.num_classes == 2 else F.softmax(y_logits, dim=1)
        y_true = y.view((-1, 1)).type_as(x) if self.num_classes == 2 else y

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits) if self.num_classes == 2 else F.softmax(y_logits, dim=1)
        y_true = y.view((-1, 1)).type_as(x) if self.num_classes == 2 else y

        # 2. Update the accuracy metric
        self.val_accuracy.update(y_scores, y_true.int())

        return y_scores, y_true

    def on_validation_epoch_end(self):
        # Compute the accuracy for the entire epoch
        val_acc = self.val_accuracy.compute()
        
        # Log the accuracy
        self.log("val_acc", val_acc, prog_bar=True)

        # Reset the metric for the next epoch
        self.val_accuracy.reset()

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        log.info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        

class FasterRCNN(LightningModule):
    def __init__(
        self,
        num_classes: int = 21,  # Pascal VOC has 20 classes + background
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        milestones: tuple = (2, 4),
        **kwargs,
    ) -> None:
        """FasterRCNN.

        Args:
            num_classes: Number of classes in the dataset (including background)
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
            milestones: List of epoch milestones for learning rate reduction
        """
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.milestones = milestones
        self.map_metric = MeanAveragePrecision()

        # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone or similar
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

        # Get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one (num_classes is the user-defined number of classes)
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Format the predictions and targets for the metric
        formatted_preds = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        formatted_targets = [{k: v.cpu() for k, v in target.items()} for target in targets]

        # Update the metric
        self.map_metric.update(formatted_preds, formatted_targets)

        return predictions

    def on_validation_epoch_end(self):
        # Compute the metric
        map_result = self.map_metric.compute()
        
        # Log the results
        self.log('val_map', map_result['map'], prog_bar=True)
        self.log('val_map_50', map_result['map_50'], prog_bar=True)
        self.log('val_map_75', map_result['map_75'], prog_bar=True)

        # Reset the metric for the next epoch
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class GPT2FineTuner(LightningModule):
    def __init__(self, lr=5e-5, num_train_steps=10000, num_warmup_steps=500):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lr = lr
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'step',
                'frequency': 1
            }
        }