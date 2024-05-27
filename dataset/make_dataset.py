import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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
    # Generate positive negative data
    n_train_samples_par_c = n_train_labeled + n_train_unlabeled
    pos_data = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu, mu])
    neg_data = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([-mu, -mu])
    # Supervised learning
    if n_train_unlabeled == 0:
        train_data = np.concatenate((pos_data, neg_data), axis=0)
        # Generate labels
        train_labels = np.zeros(2*n_train_samples_par_c, dtype=np.int64)
        train_labels[:n_train_samples_par_c] = np.array(0, dtype=np.int64)
        train_labels[n_train_samples_par_c:] = np.array(1, dtype=np.int64)
        # plot positive negative
        plt.scatter(train_data[:n_train_samples_par_c, 0],
                    train_data[:n_train_samples_par_c, 1], color=cmap(0), s=5)
        plt.scatter(train_data[n_train_samples_par_c:, 0],
                    train_data[n_train_samples_par_c:, 1], color=cmap(1), s=5)
        plt.savefig(save_dir + "/" + "train_data.png")
        plt.close()
    else:
        # Generate positive data
        if extreme:
            pos_order = np.argsort(pos_data[:, 1])
            neg_order = np.argsort(neg_data[:, 1])
        else:
            pos_order = np.array(range(n_train_samples_par_c))
            neg_order = np.array(range(n_train_samples_par_c))
        pos_labeled = pos_data[pos_order[:n_train_labeled], :]
        pos_unlabeled = pos_data[pos_order[n_train_labeled:], :]
        # Generate negative data
        neg_unlabeled = neg_data[neg_order[:n_train_unlabeled], :]
        neg_labeled = neg_data[neg_order[n_train_unlabeled:], :]
        # Concatenation
        labeled_data = np.concatenate((pos_labeled, neg_labeled), axis=0)
        unlabeled_data = np.concatenate((pos_unlabeled, neg_unlabeled), axis=0)
        # Generate labeles
        obs_labels = np.zeros(2*n_train_labeled, dtype=np.int64)
        obs_labels[:n_train_labeled] = np.array(0, dtype=np.int64)
        obs_labels[n_train_labeled:] = np.array(1, dtype=np.int64)
        unobs_labels = np.zeros(2*n_train_unlabeled, dtype=np.int64)
        unobs_labels[:n_train_unlabeled] = np.array(0, dtype=np.int64)
        unobs_labels[n_train_unlabeled:] = np.array(1, dtype=np.int64)

        # Plot positive negative
        plt.scatter(unlabeled_data[:n_train_unlabeled, 0],
                    unlabeled_data[:n_train_unlabeled, 1], color=cmap(0), facecolor='None', s=20, alpha=0.7)
        plt.scatter(labeled_data[:n_train_labeled, 0],
                    labeled_data[:n_train_labeled, 1], color=cmap(0), s=20)
        plt.scatter(unlabeled_data[n_train_unlabeled:, 0],
                    unlabeled_data[n_train_unlabeled:, 1], color=cmap(1), facecolor='None', s=20, alpha=0.7)
        plt.scatter(labeled_data[n_train_labeled:, 0],
                    labeled_data[n_train_labeled:, 1], color=cmap(1), s=20)
        plt.savefig(save_dir + "/" + "train_data.png")
        plt.close()

    # Generate test data
    pos_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu, mu])
    neg_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([-mu, -mu])
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
    else:
        print(f'train lbl: {len(labeled_data)}, unlbl:{len(unlabeled_data)}, test:{len(test_data)}')
        np.save(save_dir + "/" + "unlabeled_data.npy", unlabeled_data)
        np.save(save_dir + "/" + "labeled_data.npy", labeled_data)
        np.save(save_dir + "/" + "obs_labels.npy", obs_labels)
        np.save(save_dir + "/" + "unobs_labels.npy", unobs_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True

def make_unidimensionally_gauss_toydata(n_train_samples_par_c=1000, n_test_samples_par_c=1000, n_dim=1, mu=1.0, sigma=0.5, save_dir='1'):
    os.makedirs(save_dir, exist_ok=True)
    # Generate positive negative data
    pos_data = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu])
    neg_data = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu*-1])
    train_data = np.concatenate((pos_data, neg_data), axis=0)
    # Generate labels
    train_labels = np.zeros(2*n_train_samples_par_c, dtype=np.int64)
    train_labels[:n_train_samples_par_c] = np.array(0, dtype=np.int64)
    train_labels[n_train_samples_par_c:] = np.array(1, dtype=np.int64)

    # Plot
    cmap = get_cmap("Set1")
    # positive negative
    plt.scatter(train_data[:n_train_samples_par_c], [0]*len(train_data[:n_train_samples_par_c]), color=cmap(0), s=5)
    plt.scatter(train_data[n_train_samples_par_c:, 0], [0]*len(train_data[n_train_samples_par_c:]), color=cmap(1), s=5)
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.close()

    # Generate test data
    pos_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu])
    neg_test_data = sigma * np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu*-1])
    test_data = np.concatenate((pos_test_data, neg_test_data), axis=0)
    test_labels = np.zeros(2*n_test_samples_par_c, dtype=np.int64)
    test_labels[:n_test_samples_par_c] = np.array(0, dtype=np.int64)
    test_labels[n_test_samples_par_c:] = np.array(1, dtype=np.int64)
    # Plot positive negative
    plt.scatter(test_data[:n_test_samples_par_c], [0]*len(test_data[:n_test_samples_par_c]), color=cmap(0), s=5)
    plt.scatter(test_data[n_test_samples_par_c:], [0]*len(test_data[n_test_samples_par_c:]), color=cmap(1), s=5)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()
    # Save data
    np.save(save_dir + "/" + "train_data.npy", train_data)
    np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True

def make_sklearn_datasets(n_train_samples_par_c, data= None, noise=1.0, save_dir=None):
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles
    os.makedirs(save_dir, exist_ok=True)
    cmap = get_cmap("Set1")
    if data == 'roll':
        x, _ = make_swiss_roll(n_train_samples_par_c, noise=noise)
        train_data = x[:, [0, 2]] / 10.0
        train_labels = np.zeros(n_train_samples_par_c, dtype=np.int64)
        test_data = x[:, [0, 2]] / 10.0
        test_labels = np.zeros(n_train_samples_par_c, dtype=np.int64)
    elif data == 'moon':
        train_data, train_labels = make_moons(n_train_samples_par_c, noise=noise)
        test_data, test_labels = make_moons(n_train_samples_par_c, noise=noise)
        train_data = train_data - 0.5
        test_data = test_data - 0.5
    elif data == 'circles':
        train_data, train_labels = make_circles(n_train_samples_par_c, factor=0.3, noise=noise)
        test_data, test_labels = make_circles(n_train_samples_par_c, factor=0.3, noise=noise)
    # Plot
    plt.scatter(*train_data.T, color=cmap(train_labels), s=5)
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.close()
    plt.scatter(*test_data.T, color=cmap(test_labels), s=5)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()
    # Save data
    np.save(save_dir + "/" + "train_data.npy", train_data)
    np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)

    return True

def make_moon_ssl(n_train_labeled=10, n_train_unlabeled=240, n_test_samples_par_c=1000, noise=.05, extreme=True, save_dir=None):
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles
    os.makedirs(save_dir, exist_ok=True)
    cmap = get_cmap("Set1")
    n_train_samples_per_c = n_train_labeled + n_train_unlabeled
    train_data, train_labels = make_moons(n_train_samples_per_c*2, noise=noise)
    train_data = train_data - 0.5
    # class_split
    pos_data = train_data[np.argsort(train_labels)][:n_train_samples_per_c]
    neg_data = train_data[np.argsort(train_labels)][n_train_samples_per_c:]
    # Generate positive data
    if extreme:
        pos_order = np.argsort(pos_data[:, 0])
        neg_order = np.argsort(neg_data[:, 0])
    else:
        pos_order = np.array(range(n_train_samples_per_c))
        neg_order = np.array(range(n_train_samples_per_c))
    pos_labeled = pos_data[pos_order[:n_train_labeled], :]
    pos_unlabeled = pos_data[pos_order[n_train_labeled:], :]
    # Generate negative data
    neg_unlabeled = neg_data[neg_order[:n_train_unlabeled], :]
    neg_labeled = neg_data[neg_order[n_train_unlabeled:], :]
    # Concatenation
    labeled_data = np.concatenate((pos_labeled, neg_labeled), axis=0)
    unlabeled_data = np.concatenate((pos_unlabeled, neg_unlabeled), axis=0)
    # Generate labeles
    obs_labels = np.zeros(2*n_train_labeled, dtype=np.int64)
    obs_labels[:n_train_labeled] = np.array(0, dtype=np.int64)
    obs_labels[n_train_labeled:] = np.array(1, dtype=np.int64)
    unobs_labels = np.zeros(2*n_train_unlabeled, dtype=np.int64)
    unobs_labels[:n_train_unlabeled] = np.array(0, dtype=np.int64)
    unobs_labels[n_train_unlabeled:] = np.array(1, dtype=np.int64)

    # Plot positive negative
    plt.scatter(unlabeled_data[:n_train_unlabeled, 0],
                unlabeled_data[:n_train_unlabeled, 1], color=cmap(0), facecolor='None', s=20, alpha=0.7)
    plt.scatter(labeled_data[:n_train_labeled, 0],
                labeled_data[:n_train_labeled, 1], color=cmap(0), s=20)
    plt.scatter(unlabeled_data[n_train_unlabeled:, 0],
                unlabeled_data[n_train_unlabeled:, 1], color=cmap(1), facecolor='None', s=20, alpha=0.7)
    plt.scatter(labeled_data[n_train_labeled:, 0],
                labeled_data[n_train_labeled:, 1], color=cmap(1), s=20)
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.close()

    # Generate test data
    test_data, test_labels = make_moons(n_test_samples_par_c, noise=noise)
    test_data = test_data - 0.5
    plt.scatter(*test_data.T, color=cmap(test_labels), s=5)
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()

    # Save data
    if n_train_unlabeled == 0:
        print(f'train lbl: {len(train_data)}, unlbl:0, test:{len(test_data)}')
        np.save(save_dir + "/" + "train_data.npy", train_data)
        np.save(save_dir + "/" + "train_labels.npy", train_labels)
    else:
        print(f'train lbl: {len(labeled_data)}, unlbl:{len(unlabeled_data)}, test:{len(test_data)}')
        np.save(save_dir + "/" + "unlabeled_data.npy", unlabeled_data)
        np.save(save_dir + "/" + "labeled_data.npy", labeled_data)
        np.save(save_dir + "/" + "obs_labels.npy", obs_labels)
        np.save(save_dir + "/" + "unobs_labels.npy", unobs_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True

def artificial_data(save_dir, total_num=200):
    os.makedirs(save_dir, exist_ok=True)
    # -1から1までの範囲でデータを生成
    x_values = np.linspace(-1, 1, total_num)
    # y=x+1とy=x-1の値を計算
    pos_values = x_values + 1 + 0.1*np.random.randn(total_num)
    neg_values = x_values - 1 + 0.1*np.random.randn(total_num)

    pos_data = np.vstack((x_values, pos_values)).T
    pos_labeled = pos_data[:5, :]
    pos_unlabeled = pos_data[5:, :]
    neg_data = np.vstack((x_values, neg_values)).T
    neg_labeled = neg_data[-5:, :]
    neg_unlabeled = neg_data[:-5, :]
    # Concatenation
    labeled_data = np.concatenate((pos_labeled, neg_labeled), axis=0)
    unlabeled_data = np.concatenate((pos_unlabeled, neg_unlabeled), axis=0)
    # Generate labeles
    obs_labels = np.zeros(10, dtype=np.int64)
    obs_labels[:5] = np.array(0, dtype=np.int64)
    obs_labels[5:] = np.array(1, dtype=np.int64)
    unobs_labels = np.zeros(2*(total_num-5), dtype=np.int64)
    unobs_labels[:(total_num-5)] = np.array(0, dtype=np.int64)
    unobs_labels[(total_num-5):] = np.array(1, dtype=np.int64)
    # Plot
    cmap = get_cmap("Set1")
    # positive
    plt.scatter(unlabeled_data[:(total_num-5), 0],
                unlabeled_data[:(total_num-5), 1], color=cmap(0), facecolor='None')
    plt.scatter(labeled_data[:5, 0],
                labeled_data[:5, 1], color=cmap(0))
    # negative
    plt.scatter(unlabeled_data[(total_num-5):, 0],
                unlabeled_data[(total_num-5):, 1], color=cmap(1), facecolor='None')
    plt.scatter(labeled_data[5:, 0],
                labeled_data[5:, 1], color=cmap(1))
    # plt.show()
    plt.savefig(save_dir + "/" + "train_data.png")
    # plt.close()
    # Save data
    np.save(save_dir + "/" + "unlabeled_data.npy", unlabeled_data)
    np.save(save_dir + "/" + "labeled_data.npy", labeled_data)
    np.save(save_dir + "/" + "test_data.npy", np.vstack((labeled_data, unlabeled_data)))
    np.save(save_dir + "/" + "obs_labels.npy", obs_labels)
    np.save(save_dir + "/" + "unobs_labels.npy", unobs_labels)
    np.save(save_dir + "/" + "test_labels.npy", np.hstack((obs_labels, unobs_labels)))
    return True

if __name__ == '__main__':
    for i in range(2):
        np.random.seed(seed=i)
        # Make Gaussiun toy
        # make_gauss_toydata(n_train_labeled=5, n_train_unlabeled=45, n_test_samples_par_c=500, sigma=1.0, save_dir='./2024/ssl_extremely_gaussian/100/' + str(i))
        # make_gauss_toydata(n_train_labeled=10, n_train_unlabeled=240, n_test_samples_par_c=500, sigma=1.0, mu=4, extreme=True, save_dir='./2024/ssl_extremely_gaussian/mu4/500/' + str(i))
        # make_gauss_toydata(n_train_labeled=500, n_train_unlabeled=0, n_test_samples_par_c=500, sigma=1.0, save_dir='./2024/sl_gaussian/1000/' + str(i))
        make_gauss_toydata(n_train_labeled=1000, n_train_unlabeled=0, n_test_samples_par_c=500, mu=3 ,sigma=1.0, save_dir='./2024/sl_gaussian/1000/tkzk' + str(i))


        # make_unidimensionally_gauss_toydata(save_dir="./unidimensionally_Gaussian/" + str(i), sigma=1.0, mu=5.0)
        # make_swiss_roll(10**4, save_dir="./swiss_roll/" + str(i))
        # make_gauss_multi_toydata(save_dir="./multiclass/mu_5/" + str(i))
        # artificial_data(save_dir='./artificial/200')
        # make_sklearn_datasets(100, data="moon", save_dir="./moons/" + str(i), noise=0.1)
        # make_moon_ssl(save_dir="./moons/ssl/" + str(i+2), extreme=False)
        # make_sklearn_datasets(1000, data="circles", save_dir="./circles/" + str(i), noise=0.1)

'''
def make_gauss_toydata_ssl(n_labeled_samples=5, n_unlabeled_samples=10, n_test_samples=500, n_dim=2, sigma=0.5, save_dir='1'):
    os.makedirs(save_dir, exist_ok=True)
    n_train_samples = n_labeled_samples + n_unlabeled_samples
    # Generate positive data
    pos_data = sigma*np.random.randn(n_train_samples, n_dim) + np.array([1.0, 0.0])
    pos_order = np.argsort(pos_data[:, 1])
    pos_labeled = pos_data[pos_order[:n_labeled_samples], :]
    pos_unlabeled = pos_data[pos_order[n_labeled_samples:], :]
    # Generate negative data
    neg_data = sigma*np.random.randn(n_train_samples, n_dim) + np.array([-1.0, 0.0])
    neg_order = np.argsort(neg_data[:, 1])
    neg_unlabeled = neg_data[neg_order[:n_unlabeled_samples], :]
    neg_labeled = neg_data[neg_order[n_unlabeled_samples:], :]
    # Concatenation
    labeled_data = np.concatenate((pos_labeled, neg_labeled), axis=0)
    unlabeled_data = np.concatenate((pos_unlabeled, neg_unlabeled), axis=0)
    # Generate labeles
    obs_labels = np.zeros(2*n_labeled_samples, dtype=np.int64)
    obs_labels[:n_labeled_samples] = np.array(0, dtype=np.int64)
    obs_labels[n_labeled_samples:] = np.array(1, dtype=np.int64)
    unobs_labels = np.zeros(2*n_unlabeled_samples, dtype=np.int64)
    unobs_labels[:n_unlabeled_samples] = np.array(0, dtype=np.int64)
    unobs_labels[n_unlabeled_samples:] = np.array(1, dtype=np.int64)

    # Plot
    cmap = get_cmap("Set1")
    # positive
    plt.scatter(unlabeled_data[:n_unlabeled_samples, 0],
                unlabeled_data[:n_unlabeled_samples, 1], color=cmap(0), facecolor='None', s=20, alpha=0.7)
    plt.scatter(labeled_data[:n_labeled_samples, 0],
                labeled_data[:n_labeled_samples, 1], color=cmap(0), s=20)
    # negative
    plt.scatter(unlabeled_data[n_unlabeled_samples:, 0],
                unlabeled_data[n_unlabeled_samples:, 1], color=cmap(1), facecolor='None', s=20, alpha=0.7)
    plt.scatter(labeled_data[n_labeled_samples:, 0],
                labeled_data[n_labeled_samples:, 1], color=cmap(1), s=20)
    # plt.show()
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.close()

    # Generate test data
    pos_test_data = sigma * np.random.randn(n_test_samples, n_dim) + np.array([1.0, 0.0])
    neg_test_data = sigma * np.random.randn(n_test_samples, n_dim) + np.array([-1.0, 0.0])
    test_data = np.concatenate((pos_test_data, neg_test_data), axis=0)
    test_labels = np.zeros(2*n_test_samples, dtype=np.int64)
    test_labels[:n_test_samples] = np.array(0, dtype=np.int64)
    test_labels[n_test_samples:] = np.array(1, dtype=np.int64)
    # Plot
    # positive
    plt.scatter(test_data[:n_test_samples, 0],
                test_data[:n_test_samples, 1], color=cmap(0))
    # negative
    plt.scatter(test_data[n_test_samples:, 0],
                test_data[n_test_samples:, 1], color=cmap(1))
    # plt.show()
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()

    print(f'train lbl: {len(labeled_data)}, unlbl:{len(unlabeled_data)}, test:{len(test_data)}')
    # Save data
    np.save(save_dir + "/" + "unlabeled_data.npy", unlabeled_data)
    np.save(save_dir + "/" + "labeled_data.npy", labeled_data)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "obs_labels.npy", obs_labels)
    np.save(save_dir + "/" + "unobs_labels.npy", unobs_labels)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True

def make_gauss_multi_toydata(n_train_samples_par_c=1000, n_test_samples_par_c=1000, n_dim=2, sigma=0.5, save_dir='1'):
    os.makedirs(save_dir, exist_ok=True)
    mu_x = 5.0
    mu_y = 5.0
    data_c1 = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu_x, mu_y])
    data_c2 = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([mu_x, -mu_y])
    data_c3 = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([-mu_x, mu_y])
    data_c4 = sigma*np.random.randn(n_train_samples_par_c, n_dim) + np.array([-mu_x, -mu_y])

    # Concatenation
    train_data = np.concatenate((data_c1, data_c2, data_c3, data_c4), axis=0)
    # Generate labels
    train_labels = np.zeros(4*n_train_samples_par_c, dtype=np.int64)
    train_labels[:n_train_samples_par_c] = np.array(0, dtype=np.int64)
    train_labels[n_train_samples_par_c:2*n_train_samples_par_c] = np.array(1, dtype=np.int64)
    train_labels[2*n_train_samples_par_c:3*n_train_samples_par_c] = np.array(2, dtype=np.int64)
    train_labels[3*n_train_samples_par_c:] = np.array(3, dtype=np.int64)

    # Plot
    # cmap = get_cmap("tab20")
    cmap = get_cmap("Set1")
    plt.scatter(train_data[:n_train_samples_par_c, 0], train_data[:n_train_samples_par_c, 1], color=cmap(0), s=5)
    plt.scatter(train_data[n_train_samples_par_c:2*n_train_samples_par_c, 0],train_data[n_train_samples_par_c:2*n_train_samples_par_c, 1], color=cmap(1), s=5)
    plt.scatter(train_data[2*n_train_samples_par_c:3*n_train_samples_par_c, 0],train_data[2*n_train_samples_par_c:3*n_train_samples_par_c, 1], color=cmap(2), s=5)
    plt.scatter(train_data[3*n_train_samples_par_c:, 0],train_data[3*n_train_samples_par_c:, 1], color=cmap(3), s=5)
    # plt.show()
    plt.savefig(save_dir + "/" + "train_data.png")
    plt.close()

    # Generate test data
    t_data_c1 = sigma*np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu_x, mu_y])
    t_data_c2 = sigma*np.random.randn(n_test_samples_par_c, n_dim) + np.array([mu_x, -mu_y])
    t_data_c3 = sigma*np.random.randn(n_test_samples_par_c, n_dim) + np.array([-mu_x, mu_y])
    t_data_c4 = sigma*np.random.randn(n_test_samples_par_c, n_dim) + np.array([-mu_x, -mu_y])
    test_data = np.concatenate((t_data_c1, t_data_c2, t_data_c3, t_data_c4), axis=0)
    test_labels = np.zeros(4*n_test_samples_par_c, dtype=np.int64)
    test_labels[:n_test_samples_par_c] = np.array(0, dtype=np.int64)
    test_labels[n_test_samples_par_c:2*n_test_samples_par_c] = np.array(1, dtype=np.int64)
    test_labels[2*n_test_samples_par_c:3*n_test_samples_par_c] = np.array(2, dtype=np.int64)
    test_labels[3*n_test_samples_par_c:] = np.array(3, dtype=np.int64)
    # Plot
    plt.scatter(test_data[:n_test_samples_par_c, 0], train_data[:n_test_samples_par_c, 1], color=cmap(0), s=5)
    plt.scatter(test_data[n_test_samples_par_c:2*n_test_samples_par_c, 0],train_data[n_test_samples_par_c:2*n_test_samples_par_c, 1], color=cmap(1), s=5)
    plt.scatter(test_data[2*n_test_samples_par_c:3*n_test_samples_par_c, 0],train_data[2*n_test_samples_par_c:3*n_test_samples_par_c, 1], color=cmap(2), s=5)
    plt.scatter(train_data[3*n_test_samples_par_c:, 0],train_data[3*n_test_samples_par_c:, 1], color=cmap(3), s=5)
    # plt.show()
    plt.savefig(save_dir + "/" + "test_data.png")
    plt.close()

    # Save data
    np.save(save_dir + "/" + "train_data.npy", train_data)
    np.save(save_dir + "/" + "train_labels.npy", train_labels)
    np.save(save_dir + "/" + "test_data.npy", test_data)
    np.save(save_dir + "/" + "test_labels.npy", test_labels)
    return True
'''