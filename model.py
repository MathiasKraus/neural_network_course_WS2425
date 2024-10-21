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

class VisionClassifier(LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        milestones: tuple = (2, 4),
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_classes: int = 2,
    ):
        """VisionClassifier.

        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor.
            milestones: List of two epochs milestones.
            lr: Initial learning rate.
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone.
            num_classes: Number of classes in the dataset.
        """
        super().__init__()
        self.save_hyperparameters()

        self.backbone = backbone
        self.milestones = milestones
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_classes = num_classes

        self.__build_model()

        self.loss_func = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
        task_type = "binary" if num_classes == 2 else "multiclass"
        self.train_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=num_classes)

    def __build_model(self):
        """Define model layers & loss."""
        # Load pre-trained network:
        backbone_model = getattr(models, self.backbone)(weights="DEFAULT")
        _layers = list(backbone_model.children())[:-1]
        
        self.feature_extractor = nn.Sequential()
        for i, layer in enumerate(_layers):
            self.feature_extractor.add_module(str(i), layer)

        # Classifier:
        in_features = backbone_model.fc.in_features
        self.fc = self.__create_classifier(in_features, self.num_classes)

    def __create_classifier(self, in_features, num_classes):
        """Create the classifier layers."""
        if num_classes == 2:
            return nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

    def forward(self, x):
        """Forward pass."""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)
    
    def predict(self, x):
        self.eval()
        # Ensure no gradients are computed
        with torch.no_grad():
            y_logits = self.forward(x)
        if self.num_classes == 2:
            y_pred = torch.sigmoid(y_logits)
            y_pred = (y_pred > 0.5).int().item()
        else:
            y_pred = F.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_pred).item()
        return y_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x).squeeze()
        y_true = y.float() if self.num_classes == 2 else y
        train_loss = self.loss_func(y_logits, y_true)

        # Log the training loss
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, logger=True)

        y_pred = torch.sigmoid(y_logits) if self.num_classes == 2 else F.softmax(y_logits, dim=1)
        self.train_acc.update(y_pred, y_true)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x).squeeze()
        y_true = y.float() if self.num_classes == 2 else y
        val_loss = self.loss_func(y_logits, y_true)

        # Log the val loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

        y_pred = torch.sigmoid(y_logits) if self.num_classes == 2 else F.softmax(y_logits, dim=1)
        self.val_acc.update(y_pred, y.int())

        return {"val_loss": self.loss_func(y_logits, y_true)}

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.log("val_acc_epoch", val_acc, prog_bar=True, logger=True)
        self.val_acc.reset()

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        self.log("train_acc_epoch", train_acc, prog_bar=True, logger=True)
        self.train_acc.reset()

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
        

class ObjectDetector(LightningModule):
    def __init__(
        self,
        num_classes: int = 21,  # Pascal VOC has 20 classes + background
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        milestones: tuple = (2, 4),
    ):
        """ObjectDetector.

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

        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.val_map_metric = MeanAveragePrecision()
        self.train_loss = 0.

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        # Ensure no gradients are computed
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.train_loss += losses
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
            
        formatted_preds = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        formatted_targets = [{k: v.cpu() for k, v in target.items()} for target in targets]

        self.val_map_metric.update(formatted_preds, formatted_targets)

        return predictions

    def on_validation_epoch_end(self):
        map_result = self.val_map_metric.compute()
        self.log('val_map_epoch', map_result['map'], prog_bar=True, logger=True)
        self.val_map_metric.reset()

    def on_train_epoch_end(self):
        self.log('train_loss_epoch', self.train_loss, prog_bar=True, logger=True)
        self.train_loss = 0.

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


class GPT2(LightningModule):
    def __init__(self, lr=5e-5, num_train_steps=10000, num_warmup_steps=500):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.lr = lr
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps

    def predict(self):
        pass
    
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