import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.neighbors import KDTree

def make_gauss_toydata(n_train_labeled=1000, n_train_unlabeled=1000, n_test_samples_par_c=1000, n_dim=2, sigma=0.5, mu=2.0, extreme=True, save_dir='1'):
    """等方性ガウシアンから2クラスの二次元データセットを作成

    Args:
        n_train_labeled (int, optional): クラス毎のラベル付きデータ数. Defaults to 1000.
        n_train_unlabeled (int, optional): クラス毎のラベルなしデータ数. Defaults to 1000.
        n_test_samples_par_c (int, optional): クラス毎のテストデータ枚数. Defaults to 1000.
        n_dim (int, optional): 次元数（未実装）. Defaults to 2.
        sigma (float, optional): 分散. Defaults to 0.5.
        mu (float, optional): 平均. Defaults to 2.0.
        extreme (bool): 偏ったラベル付きデータ
        save_dir (str, optional): 保存ディレクトリ. Defaults to '1'.

    Returns:
        _type_: _description_
    """    
    os.makedirs(save_dir, exist_ok=True)
    cmap = get_cmap("Set1")
    # Generate positive data
    n_train_samples_par_c = n_train_labeled + n_train_unlabeled
    pos_data = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu, 0.0])
    # Supervised learning
    if n_train_unlabeled == 0:
        train_data = pos_data
        # Generate labels
        train_labels = np.zeros(n_train_samples_par_c, dtype=np.int64)
        # plot positive negative
        plt.scatter(*train_data.T, color=cmap(0), s=5)
        plt.savefig(save_dir + "/" + "train_data.png")
        plt.close()

    # Generate test data
    pos_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu, 0.0])
    neg_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([-mu, 0.0])
    test_data = np.concatenate((pos_test_data, neg_test_data), axis=0)
    test_labels = np.zeros(2*n_test_samples_par_c, dtype=np.int64)
    test_labels[:n_test_samples_par_c] = np.array(0, dtype=np.int64)
    test_labels[n_test_samples_par_c:] = np.array(1, dtype=np.int64)
    # Plot
    # positive
    plt.scatter(test_data[:n_test_samples_par_c, 0],
                test_data[:n_test_samples_par_c, 1], color=cmap(0), s=5)
    # negative
    plt.scatter(test_data[n_test_samples_par_c:, 0],
                test_data[n_test_samples_par_c:, 1], color=cmap(1), s=5)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()

    # Save data
    if n_train_unlabeled == 0:
        print(f'train lbl: {len(train_data)}, unlbl:0, test:{len(test_data)}')
        np.save(save_dir + "/" + "train_data.npy", train_data)
        np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True

def make_sklearn_datasets(n_samples, data= None, noise=1.0, save_dir=None, center=None, teigi=None):
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs
    os.makedirs(save_dir, exist_ok=True)
    cmap = get_cmap("Set1")
    # same as train=test (check)
    if data == 'roll':
        x, _ = make_swiss_roll(n_samples, noise=noise)
        train_data = x[:, [0, 2]] / 10.0
        test_data = x[:, [0, 2]] / 10.0
    elif data == 'moons':
        train_data, _ = make_moons(n_samples, noise=noise)
        test_data = train_data
        train_data = (train_data - 0.5)*2
        test_data = (test_data - 0.5)*2
    elif data == 'circles':
        train_data, _ = make_circles(n_samples, factor=0.3, noise=noise)
        test_data = train_data
    elif data == 'blobs':
        train_data, _ = make_blobs(random_state=2, n_samples=n_samples, n_features=2, cluster_std=.3, centers=center, center_box=(3,-3))
        test_data = train_data
    '''テストデータの周りにoutdistribution用の格子状データを配置'''
    xy_lim = 5
    X, Y = np.meshgrid(np.linspace(-xy_lim, xy_lim, 100), (np.linspace(-xy_lim, xy_lim, 100)))
    xy = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 100), np.linspace(-xy_lim, xy_lim, 100)), axis=-1).reshape(-1, 2)
    # 格子点から最も近いmoonデータ点までの距離をKDTreeを使用して計算
    distances, _ = KDTree(test_data).query(xy, k=1)
    # 距離の閾値を設定し、その閾値よりも大きい距離の格子点のみをクラス1として採用
    distance_threshold = 0.1
    grid_points_filtered = np.array(xy[distances.flatten() > distance_threshold])

    # ラベルを用意
    train_labels = np.zeros(n_samples, dtype=np.int64)
    ood_labels = np.ones(grid_points_filtered.shape[0], dtype=np.int64)

    # Plot
    '''
    plt.scatter(*train_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.savefig(save_dir + "/" + "train_data.pdf")
    '''
    plt.xlim(-teigi, teigi)
    plt.ylim(-teigi, teigi)
    plt.scatter(*train_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "train_data_show.pdf")
    plt.close()
    '''
    plt.scatter(*test_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.scatter(*grid_points_filtered.T, color=cmap(ood_labels), s=1)
    plt.savefig(save_dir + "/" + "test_data_OOD.pdf")
    plt.close()
    '''
    # Save data
    '''
    np.save(save_dir + "/" + "train_data.npy", train_data)
    np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data_ood.npy", np.vstack([test_data, grid_points_filtered]))
    np.save(save_dir + "/" + "test_labels_ood.npy", np.concatenate([train_labels, ood_labels], 0))
    '''
    return True

def make_eight_circles(n_samples, noise_std=0.1, radius=1.0, save_dir='output', distance_threshold=0.1, teigi=2):
    os.makedirs(save_dir, exist_ok=True)
    cmap = get_cmap("Set1")

    # 円周上の点を生成
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # ノイズを追加
    noise_x = np.random.normal(0, noise_std, size=(n_samples, 100))
    noise_y = np.random.normal(0, noise_std, size=(n_samples, 100))

    # 円周上の点の周りにノイズを加えた点を生成
    x_noise = x[:, None] + noise_x
    y_noise = y[:, None] + noise_y
    train_data = np.hstack((x_noise.reshape(-1, 1), y_noise.reshape(-1, 1)))
    test_data = train_data.copy()

    '''テストデータの周りにoutdistribution用の格子状データを配置'''
    xy_lim = 5
    xy = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 100), np.linspace(-xy_lim, xy_lim, 100)), axis=-1).reshape(-1, 2)
    
    # 格子点から最も近いトレーニングデータ点までの距離をKDTreeを使用して計算
    distances, _ = KDTree(test_data).query(xy, k=1)
    
    # 距離の閾値を設定し、その閾値よりも大きい距離の格子点のみをクラス1として採用
    grid_points_filtered = np.array(xy[distances.flatten() > distance_threshold])

    # ラベルを用意
    train_labels = np.zeros(train_data.shape[0], dtype=np.int64)
    ood_labels = np.ones(grid_points_filtered.shape[0], dtype=np.int64)

    # Plot
    plt.scatter(*train_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.savefig(save_dir + "/" + "train_data.pdf")
    plt.xlim(-teigi, teigi)
    plt.ylim(-teigi, teigi)
    plt.scatter(*train_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "train_data_show.pdf")
    plt.close()
    plt.scatter(*test_data.T, color=cmap(train_labels), s=3)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.scatter(*grid_points_filtered.T, color=cmap(ood_labels), s=1)
    plt.savefig(save_dir + "/" + "test_data_OOD.pdf")
    plt.close()
    # Save data
    np.save(save_dir + "/" + "train_data.npy", train_data)
    np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data_ood.npy", np.vstack([test_data, grid_points_filtered]))
    np.save(save_dir + "/" + "test_labels_ood.npy", np.concatenate([train_labels, ood_labels], 0))
    return True


if __name__ == '__main__':
    # for i in range(2):
        # np.random.seed(seed=i)
        # make_sklearn_datasets(1000, data="moons", noise=.05, save_dir='./moon/noise05_seed' + str(i), teigi=5)
        # make_sklearn_datasets(1000, data="moons", noise=.10, save_dir='./moon/noise10_seed' + str(i), teigi=5)
        # make_sklearn_datasets(1000, data="circles", noise=.05, save_dir='./circles/noise05_seed' + str(i), teigi=2)
        # make_sklearn_datasets(1000, data="circles", noise=.10, save_dir='./circles/noise10_seed' + str(i), teigi=2)
        # make_sklearn_datasets(1000, data="roll", noise=.5, save_dir='./roll/noise05_seed' + str(i), teigi=2)
        # make_sklearn_datasets(1000, data="roll", noise=1.0, save_dir='./roll/noise10_seed' + str(i), teigi=2)
        # make_sklearn_datasets(1000, data="blobs", save_dir='./blobs/std03_seed' + str(i), center=5, teigi=4)
        # make_circles(800, save_dir='./small_circles/seed' + str(i), center=5, teigi=4)
    for i in range(11):
        np.random.seed(seed=0)
        std_list=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        make_eight_circles(n_samples=8, noise_std=std_list[i], radius=1.5, save_dir='./small_circles/sigma' + str(i+1)+'_'+str(std_list[i]))