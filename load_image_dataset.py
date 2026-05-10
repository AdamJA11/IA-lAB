import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, mapping_path="image_data/mapping.txt"):
        
        with open(images_path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)

        
        with open(labels_path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        #
        self.mapping = {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    idx_str, ascii_str = line.split(" ")
                    idx = int(idx_str)
                    ascii_val = int(ascii_str)
                    self.mapping[idx] = chr(ascii_val)
        except FileNotFoundError:
            raise FileNotFoundError(f"Mapping file no found: {mapping_path}")

        
        images = np.transpose(images, (0, 2, 1)).copy()

        
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if not self.mapping:
            raise ValueError("Empty mapping cannot map label indices to character")
        self.char_labels = [self.mapping[int(l)] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), self.char_labels[idx]

    def show_sample(self, idx=0, count=1):
        if count == 1:
            img, lbl = self[idx]
            img_np = img.squeeze(0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(img_np, cmap='gray')
            plt.title(f"Label: {lbl}")
            plt.axis('off')
            plt.show()
            return

        count = min(count, len(self)-idx)
        cols = min(5, count)
        rows = (count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
        axes = np.array(axes).reshape(-1)

        for i in range(count):
            img, lbl = self[idx + i]
            img_np = img.squeeze(0).numpy()
            ax = axes[i]
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f"{idx+i}:{lbl}", fontsize=10)
            ax.axis('off')

        for i in range(count, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()




class NaiveOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(NaiveOCRModel, self).__init__()
        
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x):
        
        mean_feature = x.mean(dim=[1, 2, 3]).unsqueeze(1)
        std_feature = x.std(dim=[1, 2, 3]).unsqueeze(1)
        
        
        features = torch.cat((mean_feature, std_feature), dim=1)
        
        
        out = self.fc(features)
        return out


class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, char_to_idx, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, string_labels in train_loader:
            
            numeric_labels = torch.tensor([char_to_idx[lbl] for lbl in string_labels], dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, numeric_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"   Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader, char_to_idx, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, string_labels in test_loader:
            numeric_labels = torch.tensor([char_to_idx[lbl] for lbl in string_labels], dtype=torch.long)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(numeric_labels.numpy())
            
    
    acc = accuracy_score(all_labels, all_preds)
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    return acc, cm, all_labels, all_preds


if __name__ == "__main__":
    # Chemin
    train_images_file = "image_data/train-images-idx3-ubyte"
    train_labels_file = "image_data/train-labels-idx1-ubyte"
    test_images_file = "image_data/test-images-idx3-ubyte"
    test_labels_file = "image_data/test-labels-idx1-ubyte"

    print("Chargement ")
    train_dataset = MyDataset(train_images_file, train_labels_file)
    test_dataset = MyDataset(test_images_file, test_labels_file)
    
    
    char_to_idx = {v: k for k, v in train_dataset.mapping.items()}
    num_classes = len(train_dataset.mapping)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    print("\n--- ENTRAÎNEMENT DU MODELE NAive (Baseline : 2 features) ---")
    naive_model = NaiveOCRModel(num_classes)
    train_model(naive_model, train_loader, char_to_idx, epochs=3)
    naive_acc, naive_cm, _, _ = evaluate_model(naive_model, test_loader, char_to_idx, num_classes)
    print(f"-> Precision Naïf : {naive_acc * 100:.2f}%")

    
    print("\n--- ENTRAÎNEMENT DU CNN (Avance) ---")
    cnn_model = OCRModel(num_classes)
    train_model(cnn_model, train_loader, char_to_idx, epochs=3)
    cnn_acc, cnn_cm, cnn_y_true, cnn_y_pred = evaluate_model(cnn_model, test_loader, char_to_idx, num_classes)
    print(f"-> Précision Modèle CNN : {cnn_acc * 100:.2f}%")

    
    print("\n--- RAPPORT DE CLASSIFICATION DU CNN ---")
   
    print(classification_report(cnn_y_true, cnn_y_pred, zero_division=0))

    
    print("\nGeneration de la matrice de confusion...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(cnn_cm, annot=False, cmap='Blues', fmt='g')
    plt.title('Matrice de Confusion - Modele CNN')
    plt.xlabel('Predictions (Ce que le modele lit)')
    plt.ylabel('Vraies Valeurs ')
    plt.show()