import numpy as np
import matplotlib.pyplot as plt


# NumPyで保存されたデータをロードする
# ここでは例として'data.npy'というファイル名を使用します
data = np.load('./exp_kl_final_pdf/logs/OURS/moon/ckpts/final_pseudo_samples_7500.npy')

# print(data)
# データが2次元の座標（x, y）のペアであると仮定します
x = data[:, 0]
y = data[:, 1]

# Matplotlibを使って散布図をプロットする
plt.scatter(x, y, color='blue', alpha=1, s=5)
# plt.xlim(-4,4)
# plt.ylim(-3,2)
plt.xlim(-6,6)
plt.ylim(-6,6)
plt.xticks([])
plt.yticks([])

# グラフを表示
plt.show()
plt.savefig('./sample_6500.pdf')