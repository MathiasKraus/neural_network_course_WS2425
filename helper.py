import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightning.pytorch.callbacks import TQDMProgressBar

#################
#### General ####
#################

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

def get_sample(data_module, val=False, idx=None):
    if val == True:
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.train_dataloader()
    if idx is None:
        idx = np.random.randint(0, len(dataloader.dataset) - 1)
    return dataloader.dataset[idx]

def get_batch(data_module, val=False):
    if val == True:
        return next(iter(data_module.val_dataloader()))
    else:
        return next(iter(data_module.train_dataloader()))

def show_confusion_matrix(model, data_module):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_module.val_dataloader():
            outputs = model(images)
            if outputs.shape[1] == 1 or outputs.dim() == 1:
                preds = torch.round(torch.sigmoid(outputs).squeeze())
            else:
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Find the most confused classes
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_confusions = cm_no_diag.max(axis=1)
    max_confusions_indices = cm_no_diag.argmax(axis=1)

    # Plot the confusion matrix
    plt.figure(figsize=(7, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=.5)

    # Highlight the most confused classes
    for i in range(len(data_module.class_names)):
        if max_confusions[i] > 0:
            ax.add_patch(plt.Rectangle((max_confusions_indices[i], i), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.set_xticklabels(data_module.class_names, rotation=45)
    ax.set_yticklabels(data_module.class_names, rotation=0)
    plt.show()


#################
#### Vision #####
#################

def show_image(x):
    """
    Plots an image from a tensor.
    
    Args:
    x (torch.Tensor): The image tensor. Can be gray (rank 2) or color (rank 3).
                      The color dimension can be the first or the last.
    """
    # Ensure the tensor is on the CPU and convert to numpy
    if x.is_cuda:
        x = x.cpu()
    x = x.numpy()

    if x.min() < 0:
        x = (x - x.min()) / (x.max() - x.min())
    
    # Handle gray images (rank 2)
    if len(x.shape) == 2:
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.show()
        return

    # Handle color images (rank 3)
    if len(x.shape) == 3:
        # If the color channel is the first dimension (C, H, W), move it to the last dimension (H, W, C)
        if x.shape[0] in [1, 3]:  # Assuming the first dimension is the color channel
            x = np.transpose(x, (1, 2, 0))
        
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        return

    raise ValueError("Input tensor must be of rank 2 (gray image) or rank 3 (color image)")

def show_image_and_bounding_box(x, bboxes=None):
    """
    Plots an image from a tensor and optionally draws bounding boxes.

    Args:
    x (torch.Tensor): The image tensor. Can be gray (rank 2) or color (rank 3).
                      The color dimension can be the first or the last.
    bboxes (list of dict): Optional. List of bounding boxes and labels, each as a dictionary with 'name' and 'bndbox'.
                           The 'bndbox' should have keys 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    # Ensure the tensor is on the CPU and convert to numpy
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    if x.min() < 0:
        x = (x - x.min()) / (x.max() - x.min())
    
    # Handle gray images (rank 2)
    if len(x.shape) == 2:
        plt.imshow(x, cmap='gray')
    # Handle color images (rank 3)
    elif len(x.shape) == 3:
        # If the color channel is the first dimension (C, H, W), move it to the last dimension (H, W, C)
        if x.shape[0] in [1, 3]:  # Assuming the first dimension is the color channel
            x = np.transpose(x, (1, 2, 0))
        
        # If it's a single-channel image, repeat to make it RGB
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
    else:
        raise ValueError("Input tensor must be of rank 2 (gray image) or rank 3 (color image)")

    # Create figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(x)

    # Draw bounding boxes if provided
    if bboxes is not None:
        for bbox in bboxes:
            box = bbox['bndbox']
            xmin = float(box['xmin'])
            ymin = float(box['ymin'])
            xmax = float(box['xmax'])
            ymax = float(box['ymax'])
            label = bbox['name']
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(xmin, ymin, label, color='r', fontweight='bold')

    plt.axis('off')
    plt.show()

def show_worst_image_predictions(model, data_module, n=5):
    model.eval()  # Set the model to evaluation mode
    misclassified = {}

    with torch.no_grad():
        for images, labels in data_module.val_dataloader():
            outputs = model(images)
            if outputs.shape[1] == 1 or outputs.dim() == 1:
                predicted = torch.sigmoid(outputs).squeeze()
                prediction_error = torch.abs(predicted - labels.float())
            else:
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted_class = torch.max(probabilities, 1)
                predicted = probabilities[range(len(labels)), labels]
                prediction_error = 1 - predicted

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                error = prediction_error[i].item()

                if true_label != pred_label:
                    if true_label not in misclassified:
                        misclassified[true_label] = []
                    misclassified[true_label].append((images[i], pred_label, error))

    # Sort and select the worst predictions
    for true_label in misclassified:
        misclassified[true_label].sort(key=lambda x: x[2], reverse=True)
        misclassified[true_label] = misclassified[true_label][:n]

    # Plot the misclassified images
    num_classes = len(misclassified)
    fig, axes = plt.subplots(num_classes, n, figsize=(15, 5 * num_classes))

    for true_label, misclassified_items in misclassified.items():
        for i, (img, pred_label, error) in enumerate(misclassified_items):
            ax = axes[true_label, i] if num_classes > 1 else axes[i]
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize image for display
            ax.imshow(img)
            ax.set_title(f"Output Neuron: {pred_label:.2f}, True: {true_label}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def convert_predictions_bboxes(predictions, idx_to_class, threshold=0.5):
    converted_predictions = []
    for pred in predictions:
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                converted_predictions.append({
                    'name': idx_to_class[label.item()],  # Assuming you have a list of class names
                    'pose': 'Unspecified',
                    'truncated': '0',
                    'occluded': '0',
                    'bndbox': {
                        'xmin': str(int(box[0].item())),
                        'ymin': str(int(box[1].item())),
                        'xmax': str(int(box[2].item())),
                        'ymax': str(int(box[3].item())),
                    },
                    'difficult': '0'
                })
    
    return converted_predictions


##################
##### Text #######
##################

def generate_text(model, tokenizer, starting_text, max_length=100):
    # Tokenize the starting text
    input_ids = tokenizer.encode(starting_text, return_tensors='pt')
    
    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    # Generate text
    with torch.no_grad():
        output = model.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text