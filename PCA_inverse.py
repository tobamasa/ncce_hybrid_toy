# MNISTをPCAで次元圧縮
# 可視化
# 特徴ベクトル持ってくる
import torch, os
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# データ変換: 画像をテンソルに変換し、0-1の範囲に正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# トレーニングデータのロード
train_dataset = datasets.MNIST(root='../ncce_hybrid/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)

# データのロード
for images, labels in train_loader:
    images = images.view(-1, 28*28)  # 画像をフラット化
    images = images.numpy()
    labels = labels.numpy()
    break  # データ全体を1つのバッチとしてロード

# # PCAモデルの保存
# joblib.dump(pca, 'pca_model.pkl')

# # 圧縮されたデータのロード
# compressed_data = np.load("test_data_pca.npy")

# # PCAモデルのロード
# pca = joblib.load('pca_model.pkl')

# PCAの適用
pca = PCA(n_components=10)
X_pca = pca.fit_transform(images)

print("Original shape:", images.shape)
print("Transformed shape:", X_pca.shape)
for i in range(47):
    epoch = (i+1)*100
    # 圧縮データのロード
    compressed_data = np.load(f"./exp_mnist/logs/mnist_bigmodel/ckpts/final_pseudo_samples_{epoch}.npy")
    # 復元
    restored_data = pca.inverse_transform(compressed_data)
    print("Restored data shape:", restored_data.shape)
    # 画像として表示
    restored_images = restored_data.reshape(-1, 28, 28)  # 28x28の画像にリシェイプ
    # 画像をテンソルに変換
    restored_images_tensor = torch.tensor(restored_images).unsqueeze(1)  # (N, 1, 28, 28)の形状に変換

    # グリッド画像の作成
    image_grid = make_grid(restored_images_tensor[:100], nrow=10, normalize=True)
    # グリッド画像の保存
    save_image(image_grid, os.path.join('./exp_mnist/logs/mnist_bigmodel/samples/', f'restored_image_grid_{epoch}.png'))

# train_id_labels = np.zeros(len(labels), dtype=np.int64)

# np.save("./" + "train_data.npy", X_pca)
# np.save("./" + "train_true_labels.npy", labels)
# np.save("./" + "train_labels.npy", train_id_labels)

# # 主成分を可視化する
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
# plt.colorbar(scatter)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('MNIST PCA Projection (First Two Components)')
# plt.savefig('./mnist_pca.png')


# 多分やり方違う？PCAが違うしな
# test_ood_dataset = datasets.FashionMNIST(root='../../ncce_hybrid/data', train=False, download=True, transform=transform)
# test_id_dataset = datasets.MNIST(root='../../ncce_hybrid/data', train=False, download=True, transform=transform)
# test_ood_loader = torch.utils.data.DataLoader(dataset=test_ood_dataset, batch_size=len(train_dataset), shuffle=False)
# test_id_loader = torch.utils.data.DataLoader(dataset=test_id_dataset, batch_size=len(train_dataset), shuffle=False)
# # データのロード
# for images_id, _ in test_id_loader:
#     images_id = images_id.view(-1, 28*28)  # 画像をフラット化
#     images_id = images_id.numpy()
#     break

# for images_ood, _ in test_ood_loader:
#     images_ood = images_ood.view(-1, 28*28)  # 画像をフラット化
#     images_ood = images_ood.numpy()
#     break

# PCAの適用
# X_id_pca = pca.transform(images_id)
# X_ood_pca = pca.transform(images_ood)
# X_combined_pca = np.concatenate((X_id_pca, X_ood_pca), axis=0)

# test_labels = np.zeros(2*len(images_id), dtype=np.int64)
# test_labels[:len(images_id)] = np.array(0, dtype=np.int64)
# test_labels[len(images_id):] = np.array(1, dtype=np.int64)

# np.save("./" + "test_data.npy", X_combined_pca)
# np.save("./" + "test_labels.npy", test_labels)

# print("Original shape:", images_id.shape)
# print("Transformed shape:", X_combined_pca.shape)
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_combined_pca[:, 0], X_combined_pca[:, 1], c=test_labels, cmap='tab10', alpha=0.5)
# plt.colorbar(scatter)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('MNIST PCA Projection (First Two Components)')
# plt.savefig('./mnist_fashion_pca.png')