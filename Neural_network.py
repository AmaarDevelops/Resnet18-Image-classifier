import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision
import torch
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import torchvision.models as models



train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(), # New Augmentation
    transforms.RandomRotation(10),     # New Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # New Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

train_dataset = ImageFolder('./data/train',transform=train_transform)

val_dataset = ImageFolder('./data/val',transform=transform)

test_dataset = ImageFolder('./data/test',transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True)


# ------------------------- Convolutional Neural Network -----------------------------

# Hyper-Parameters

learning_rate = 0.001
batch_size=32
num_epochs = 10
epochs_list = range(1,num_epochs + 1)

model = models.resnet18(weights='IMAGENET1K_V1')

# freeeze the entire backbone to accelerate training
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)

# tracking losses
train_losses = []
val_losses = []

# tracking accuracies
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0
    n_train_samples = len(train_loader.dataset)

    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Output
        output = model(images)
        # Loss calculation
        loss = criterion(output,labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss_sum += loss.item()

        # Tracking training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
          print(f'Epoch: {epoch + 1}/{num_epochs}, Step: {i + 1}/{n_total_steps}, Loss: {loss.item():.4f}')

    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_losses.append(avg_train_loss)

    # Accuracy
    avg_train_accuracy = epoch_train_n_correct / n_train_samples
    train_accuracies.append(avg_train_accuracy)

    # ... (Plot loss curve code) ...



    # Loss Calulating
    model.eval()
    val_total_loss=0.0
    val_n_total_batches = 0
    val_n_correct = 0


    with torch.no_grad():
       n_correct = 0
       n_samples = len(train_loader.dataset)

       for images,labels in val_loader:
           images,labels = images.to(device) , labels.to(device)
           output = model(images)

           loss = criterion(output,labels)
           val_total_loss += loss.item()
           val_n_total_batches += 1

           _,predicted = torch.max(output,1)
           val_n_correct += (predicted == labels).sum().item()

    avg_val_loss = val_total_loss / val_n_total_batches
    val_losses.append(avg_val_loss)

    val_n_samples = len(val_loader.dataset)
    val_accuracy = val_n_correct / val_n_samples
    val_accuracies.append(val_accuracy)

    print(f"\n✨ EPOCH {epoch + 1} SUMMARY: Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}\n")

     # Plot Accuracy Curve
    current_epochs = epochs_list[:epoch + 1] # e.g., [1], then [1, 2], then [1, 2, 3]...

    plt.figure(figsize=(10, 6))
    plt.plot(current_epochs, train_accuracies, label='Training Accuracy', color='green')
    plt.plot(current_epochs, val_accuracies, label='Validation Accuracy', color='orange', linestyle='--')

    plt.title(f'Training and Validation Accuracy Curve (Epoch {epoch + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(current_epochs) # Set ticks dynamically
    plt.show()

    
print('Finished Training')



# Plot loss curve
epochs_list = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_losses, label='Training Loss', color='blue')
plt.plot(epochs_list, val_losses, label='Validation Loss', color='red', linestyle='--')

plt.title('Training and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.xticks(epochs_list)
plt.show()


# -------------------------------- Evaluating on Test Data ------------------
model.eval()
y_true = []
y_pred = []
total_samples = len(test_loader.dataset)

with torch.no_grad():
    for images,labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        output = model(images)

        _,predicted = torch.max(output.data,1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Converting into array

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------- Calculating perfomance metrics ------------
accuracy = accuracy_score(y_true,y_pred)
print(f'Accuract of CNN :- {accuracy}')

precision = precision_score(y_true,y_pred,average='macro')
print(f'Precision of CNN :- {precision}')

recall = recall_score(y_true,y_pred,average='macro')
print(f'Recall of CNN :- {recall}')

# Confusion matrix
class_names = ['Dog','Cat']
cm = confusion_matrix(y_true,y_pred)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 7))
# Use normalized data for the heatmap
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar=False) # Turn off color bar as percentages are clear
plt.title('Normalized Confusion Matrix (Row Percentages)')
plt.ylabel('True Label (Cat/Dog)')
plt.xlabel('Predicted Label')
plt.show()


#PLotting the Confusin matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# Saving the model after training
torch.save(model.state_dict(),"dogcat_resnet18.pth")

# save the class indexing (required from deployment)
class_to_idx = train_dataset.class_to_idx
torch.save(class_to_idx,'class_to_index.pth')


print('Everything done')
