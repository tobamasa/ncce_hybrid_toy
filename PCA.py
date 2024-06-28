# MNISTをPCAで次元圧縮
# 可視化
# 特徴ベクトル持ってくる
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import joblib

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# データ変換: 画像をテンソルに変換し、0-1の範囲に正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# トレーニングデータのロード
train_dataset = datasets.MNIST(root='../../ncce_hybrid/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)

# データのロード
for images, labels in train_loader:
    images = images.view(-1, 28*28)  # 画像をフラット化
    images = images.numpy()
    labels = labels.numpy()
    break  # データ全体を1つのバッチとしてロード

# PCAの適用
pca = PCA(n_components=2)
X_pca = pca.fit_transform(images)

print("Original shape:", images.shape)
print("Transformed shape:", X_pca.shape)

train_id_labels = np.zeros(len(labels), dtype=np.int64)

np.save("./" + "train_data.npy", X_pca)
np.save("./" + "train_true_labels.npy", labels)
np.save("./" + "train_labels.npy", train_id_labels)

# 主成分を可視化する
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MNIST PCA Projection (First Two Components)')
plt.savefig('./mnist_pca.png')


# 多分やり方違う？PCAが違うしな
test_ood_dataset = datasets.FashionMNIST(root='../../ncce_hybrid/data', train=False, download=True, transform=transform)
test_id_dataset = datasets.MNIST(root='../../ncce_hybrid/data', train=False, download=True, transform=transform)
test_ood_loader = torch.utils.data.DataLoader(dataset=test_ood_dataset, batch_size=len(train_dataset), shuffle=False)
test_id_loader = torch.utils.data.DataLoader(dataset=test_id_dataset, batch_size=len(train_dataset), shuffle=False)
# データのロード
for images_id, _ in test_id_loader:
    images_id = images_id.view(-1, 28*28)  # 画像をフラット化
    images_id = images_id.numpy()
    break

for images_ood, _ in test_ood_loader:
    images_ood = images_ood.view(-1, 28*28)  # 画像をフラット化
    images_ood = images_ood.numpy()
    break

# PCAの適用
X_id_pca = pca.transform(images_id)
X_ood_pca = pca.transform(images_ood)
X_combined_pca = np.concatenate((X_id_pca, X_ood_pca), axis=0)

test_labels = np.zeros(2*len(images_id), dtype=np.int64)
test_labels[:len(images_id)] = np.array(0, dtype=np.int64)
test_labels[len(images_id):] = np.array(1, dtype=np.int64)

np.save("./" + "test_data.npy", X_combined_pca)
np.save("./" + "test_labels.npy", test_labels)

print("Original shape:", images_id.shape)
print("Transformed shape:", X_combined_pca.shape)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_combined_pca[:, 0], X_combined_pca[:, 1], c=test_labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MNIST PCA Projection (First Two Components)')
plt.savefig('./mnist_fashion_pca.png')

joblib.dump(pca, './pca_model.pkl')