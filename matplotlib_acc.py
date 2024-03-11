import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['ResNet18', 'ResNet50', 'MLP3Layer', 'MLP5Layer', 'Swin-Transformer-Tiny']

# Optimal accuracy for each model
optimal_accuracy = [94.76, 96.36, 44.27, 54.59, 96.45]

# Per-class accuracy for each model
per_class_accuracy = [
    [95, 96, 93, 89, 96, 89, 97, 96, 97, 95],  # ResNet18
    [97, 97, 95, 93, 97, 92, 96, 97, 97, 96],  # ResNet50
    [39, 41, 35, 37, 26, 41, 40, 52, 63, 59],  # MLP3Layer
    [57, 68, 31, 42, 44, 53, 56, 57, 65, 58],  # MLP5Layer
    [96, 98, 95, 92, 96, 94, 99, 97, 97, 97]   # Swin-Transformer-Tiny
]

# Openset threshold for each model
openset_threshold = [0.99997, 0.99997, 0.62569, 0.78582, 0.99999]

# Recall for each model
recall = [1, 0, 53, 21, 1]

# Plot optimal accuracy
plt.figure(figsize=(10,5))
plt.plot(models, optimal_accuracy, marker='o')
for i, v in enumerate(optimal_accuracy):
    plt.text(i, v, " "+str(v), va='center', ha='left')
plt.ylabel('Optimal Accuracy')
plt.title('Optimal Accuracy for Each Model')
plt.grid(True)
plt.savefig(f'optimal_accuracy.png')

# Plot per-class accuracy
plt.figure(figsize=(15,7))
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
per_class_accuracy = np.array(per_class_accuracy)
width = 0.15  # the width of the bars
x = np.arange(len(classes))  # the label locations

for i in range(len(models)):
    plt.bar(x - width*2 + i*width, per_class_accuracy[i, :], width, label=models[i])
    for j, v in enumerate(per_class_accuracy[i, :]):
        plt.text(j - width*2 + i*width, v, " "+str(v), va='center', ha='left')

plt.ylabel('Per-Class Accuracy')
plt.title('Per-Class Accuracy for Each Model')
plt.xticks(x, classes)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'per_class_accuracy.png')


# Plot openset threshold
plt.figure(figsize=(10,5))
plt.plot(models, openset_threshold, marker='o')
for i, v in enumerate(openset_threshold):
    plt.text(i, v, " "+str(v), va='center', ha='left')
plt.ylabel('Openset Threshold')
plt.title('Openset Threshold for Each Model')
plt.grid(True)
plt.savefig(f'openset_threshold.png')

# Plot openset recall
plt.figure(figsize=(10,5))
plt.plot(models, recall, marker='o')
for i, v in enumerate(recall):
    plt.text(i, v, " "+str(v), va='center', ha='left')
plt.ylabel('Recall')
plt.title('Recall of Openset for Each Model')
plt.grid(True)
plt.savefig(f'openset_recall.png')
