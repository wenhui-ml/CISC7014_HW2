import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import timm
from timm.models import create_model
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Define global variables for plt
pr_fig, pr_ax = plt.subplots()
roc_fig, roc_ax = plt.subplots()

# Define global variables for DL
cuda_device = "cuda:1"      # specify CUDA device here
batch_size = 32
image_size = 224
num_epochs = 20
learning_rate=0.001
num_classes_cifar10 = 10
num_classes_fashionmnist = 10

# Download and save CIFAR10 data with normalization for CNN
transform_dl = transforms.Compose([
    transforms.Resize((image_size, image_size)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_data_dl = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_dl)
test_data_dl = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_dl)

train_loaders = DataLoader(train_data_dl, batch_size=batch_size, shuffle=True)
test_loaders = DataLoader(test_data_dl, batch_size=batch_size, shuffle=False)

# Download and save FashionMNIST data with normalization for CNN
transform_dl_fashion = transforms.Compose([
    transforms.Resize((image_size, image_size)), 
    transforms.Grayscale(num_output_channels=3),  # Move this line up
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)),
])
train_data_dl_fashion = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_dl_fashion)
test_data_dl_fashion = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_dl_fashion)

train_loaders_fashion = DataLoader(train_data_dl_fashion, batch_size=batch_size, shuffle=True)
test_loaders_fashion = DataLoader(test_data_dl_fashion, batch_size=batch_size, shuffle=False)
total_data_dl_fashion = torch.utils.data.ConcatDataset([train_data_dl_fashion, test_data_dl_fashion])
open_set_loader = DataLoader(total_data_dl_fashion, batch_size=batch_size, shuffle=False)

# Function to visualize class distribution
def visualize_class_distribution(train_dataset, test_dataset, classes, title):
    class_counts_train = np.zeros(len(classes))
    class_counts_test = np.zeros(len(classes))
    for _, label in train_dataset:
        class_counts_train[label] += 1
    for _, label in test_dataset:
        class_counts_test[label] += 1

    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, class_counts_train, width, label='Train')
    rects2 = ax.bar(x + width/2, class_counts_test, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_dataset.png')

# Define function to compute metrics and plot curves
def compute_metrics_and_plot_curves(all_labels, all_predictions, task_name="", dataset_name=""):
    # Compute precision, recall, PR curve, and ROC curve here...
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions, pos_label=1)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)

    pr_ax.plot(recall, precision, lw=2, label=f'{task_name} on {dataset_name}')

    roc_ax.plot(fpr, tpr, lw=2, label=f'{task_name} on {dataset_name} (area = {roc_auc:.2f})')

# Define 3-layer MLP model
class MLP3Layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP3Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define 5-layer MLP model
class MLP5Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP5Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

# Define ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Change this line
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Define ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Change this line
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Define Swin-Transformer-Tiny model
class SwinTransformerTiny(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerTiny, self).__init__()
        self.model = create_model('swin_tiny_patch4_window7_224', pretrained=True)  # Change this line
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

def train_and_evaluate_with_model(train_loader, test_loader, num_classes, model, classes, task_name="", dataset_name="", open_set_loader=None):
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")  # use global variable here
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training code here...
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print statistics (accuracy) every epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Accuracy on Test Set: {accuracy:.2f}%')

        # deep copy the model
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save the best model weights along with task_name, epoch, and acc
            model_info = {
                'task_name': task_name,
                'epoch': epoch,
                'acc': best_acc,
                'model_state_dict': best_model_wts
            }
            torch.save(model_info, f'{task_name}_best_model_weights.pth')

    print('Finished Training')
    
    # Evaluation code here...
    all_labels = []
    all_predictions = []
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in test_loaders:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            # Save the predicted probabilities for the positive class
            all_predictions.extend(probabilities[:, 1].cpu().numpy())
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'Final Accuracy on the Test Set: {accuracy:.2f}%')
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    compute_metrics_and_plot_curves(all_labels, all_predictions, task_name=task_name, dataset_name=dataset_name)

    # Load the best model weights along with task_name, epoch, and acc
    model_info = torch.load(f'{task_name}_best_model_weights.pth')
    model.load_state_dict(model_info['model_state_dict'])
    print(f"Loaded model from {model_info['task_name']} with accuracy {model_info['acc']} at epoch {model_info['epoch']+1}")

    # Initialize counters for open set
    TP = 0  # True Positive
    FP = 0  # False Positive
    TN = 0  # True Negative
    FN = 0  # False Negative

    test_scores = []
    for data in test_loader:
        images = data[0].to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, _ = torch.max(probabilities, dim=1)
        test_scores.extend(max_probs.cpu().detach().numpy())

    # Calculate threshold, here we choose the median of test_scores
    threshold = np.median(test_scores)
    print(f'Openset Threshold for {dataset_name} test set: {threshold}')

    # Evaluate the open set
    for data in open_set_loader:
        images = data[0].to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, _ = torch.max(probabilities, dim=1)
        # If the score exceeds the threshold, then the model is considered to correctly predict that it belongs to the open set, this is a true positive
        TP += torch.sum(max_probs.cpu().detach() > threshold).item()
        # If the score does not exceed the threshold, then the model is considered to incorrectly predict that it does not belong to the open set, this is a false negative
        FN += torch.sum(max_probs.cpu().detach() <= threshold).item()

    # Calculate evaluation metrics
    recall = TP / (TP + FN)

    print(f'Open Set Recognition Evaluation:')
    print(f'Recall: {recall:.2f}')

def main():
    # Visualize class distribution of Cifar10 and FashionMNIST
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fashionmnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    visualize_class_distribution(train_data_dl, test_data_dl, cifar10_classes, 'Class Distribution in CIFAR10 Datasets')
    visualize_class_distribution(train_data_dl_fashion, test_data_dl_fashion, fashionmnist_classes, 'Class Distribution in FashionMNIST Datasets')

    # Train and evaluate resnet18 with Cifar10
    print("\nTraining and evaluating with CNN (resnet18) on cifar10:")
    resnet18_model_1 = ResNet18(num_classes_cifar10)
    train_and_evaluate_with_model(train_loaders, test_loaders, num_classes_cifar10, resnet18_model_1, cifar10_classes, task_name="ResNet18", dataset_name="cifar10", open_set_loader=open_set_loader)

    # Train and evaluate resnet50 with Cifar10
    print("\nTraining and evaluating with CNN (resnet50) on cifar10:")
    resnet50_model_1 = ResNet50(num_classes_cifar10)
    train_and_evaluate_with_model(train_loaders, test_loaders, num_classes_cifar10, resnet50_model_1, cifar10_classes, task_name="ResNet50", dataset_name="cifar10", open_set_loader=open_set_loader)

    # Train and evaluate with 3-layer MLP on Cifar10
    print("\nTraining and evaluating with 3-layer MLP on cifar10:")
    mlp3layer_model_1 = MLP3Layer(image_size*image_size*3, 500, num_classes_cifar10)  # assuming input size is image_size*image_size*3
    train_and_evaluate_with_model(train_loaders, test_loaders, num_classes_cifar10, mlp3layer_model_1, cifar10_classes, task_name="MLP3Layer", dataset_name="cifar10", open_set_loader=open_set_loader)

    # Train and evaluate with 5-layer MLP on Cifar10
    print("\nTraining and evaluating with 5-layer MLP on cifar10:")
    mlp5layer_model_1 = MLP5Layer(image_size*image_size*3, 500, 500, 500, num_classes_cifar10)  # assuming input size is image_size*image_size*3
    train_and_evaluate_with_model(train_loaders, test_loaders, num_classes_cifar10, mlp5layer_model_1, cifar10_classes, task_name="MLP5Layer", dataset_name="cifar10", open_set_loader=open_set_loader)

    # Train and evaluate with Swin-Transformer-Tiny on Cifar10
    print("\nTraining and evaluating with Swin-Transformer-Tiny on cifar10:")
    swin_transformer_model_1 = SwinTransformerTiny(num_classes_cifar10)
    train_and_evaluate_with_model(train_loaders, test_loaders, num_classes_cifar10, swin_transformer_model_1,  cifar10_classes, task_name="Swin-Transformer-Tiny", dataset_name="cifar10", open_set_loader=open_set_loader)

    # 设置PR图形的属性并保存
    pr_ax.set_xlim([0.0, 1.0])
    pr_ax.set_ylim([0.0, 1.05])
    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title('Precision-Recall curve')
    pr_ax.legend(loc="lower right")
    pr_fig.savefig('PR_curve.png')

    # 设置ROC图形的属性并保存
    roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('Receiver Operating Characteristic')
    roc_ax.legend(loc="lower right")
    roc_fig.savefig('ROC_curve.png')

if __name__ == "__main__":
    main()