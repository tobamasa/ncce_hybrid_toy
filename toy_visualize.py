import numpy as np
import torch
import torch.nn as nn
import argparse, random, os
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import shutil, sys
from torch.utils.data import DataLoader, TensorDataset
from toy_ncsn_runner import JointLearningModel

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Parser
parser = argparse.ArgumentParser(description="Visualization t-Sne or PCA")
# parser.add_argument("--load_directory", type=str, default='./exp/logs/2dim/moon_jl_clfwonoise/best_acc_model.pth')
# parser.add_argument("--load_directory", type=str, default='./exp/logs/2dim/ssl/mu_1/5samples_jl2/ckpts/ckpt_model_4900.pth')
# parser.add_argument("--save_directory", type=str, default='./exp/logs/2dim/ssl/mu_1/5samples_jl2/ckpt_4900/')
parser.add_argument("--load_directory", type=str, default='./exp/logs/2dim/ssl/artificial/500/ckpts/ckpt_model_4900.pth')
parser.add_argument("--save_directory", type=str, default='./exp/logs/2dim/ssl/artificial/500/ckpt_4900/')
parser.add_argument("--vis_tools", type=str, default='tsne', choices=['tsne', 'pca'])
parser.add_argument("--save_feature", action="store_true")
args = parser.parse_args()
assert args.load_directory != None, "Set load_directory!"
assert args.save_directory != None, "Set save_directory!"

# Make savefolder
if os.path.exists(args.save_directory):
	overwrite = False
	response = input("Folder already exists. Overwrite? (Y/N)")
	if response.upper() == 'Y':
		shutil.rmtree(args.save_directory)
		os.makedirs(args.save_directory)
	else:
		print("Folder exists. Program halted.")
		sys.exit(0)
else:
    os.makedirs(args.save_directory)

# Dataset
if False:
	# train_data = np.load(f"./dataset/moons/0/train_data.npy")
	# train_labels = np.load(f"./dataset/moons/0/train_labels.npy")
	# test_data = np.load(f"./dataset/moons/0/test_data.npy")
	# test_labels = np.load(f"./dataset/moons/0/test_labels.npy")
	labeled_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
	dload_train = DataLoader(labeled_dataset, batch_size=10000, shuffle=False, drop_last=False)
	test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
	dload_test = DataLoader(test_dataset, batch_size=10000, shuffle=False, drop_last=False)
else:
	# labeled_data = np.load(f"./dataset/isotropic_Gaussian/ssl/mu_1/5samples/labeled_data.npy")
	# labeled_labels = np.load(f"./dataset/isotropic_Gaussian/ssl/mu_1/5samples/obs_labels.npy")
	# unlabeled_data = np.load(f"./dataset/isotropic_Gaussian/ssl/mu_1/5samples/unlabeled_data.npy")
	# unlabeled_labels = np.load(f"./dataset/isotropic_Gaussian/ssl/mu_1/5samples/unobs_labels.npy")
	labeled_data = np.load(f"./dataset/artificial/500/labeled_data.npy")
	labeled_labels = np.load(f"./dataset/artificial/500/obs_labels.npy")
	unlabeled_data = np.load(f"./dataset/artificial/500/unlabeled_data.npy")
	unlabeled_labels = np.load(f"./dataset/artificial/500/unobs_labels.npy")	
	labeled_dataset = TensorDataset(torch.from_numpy(labeled_data).float(), torch.from_numpy(labeled_labels).long())
	unlabeled_dataset = TensorDataset(torch.from_numpy(unlabeled_data).float(), torch.from_numpy(unlabeled_labels).long())
	dload_train = DataLoader(labeled_dataset, batch_size=10000, shuffle=False, drop_last=False)
	dload_unlabeled = DataLoader(unlabeled_dataset, batch_size=10000, shuffle=False, drop_last=False)
n_classes = 2
n_channels = 1
print(f'Data samples: {len(labeled_dataset)}')
print(f'Data samples: {len(unlabeled_dataset)}')

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Model
f = JointLearningModel(2, 4, 2)
print(f"loading model from {args.load_directory}")
ckpt_dict = torch.load(f"{args.load_directory}")

# Load Model weights
f.load_state_dict(ckpt_dict['model_state_dict'], strict=True)
f.lin3 = nn.Sequential()
f = f.to(device)

# feature extract
gts, features, posts = [], [], []
for _, (x_p_d, y_p_d) in tqdm(enumerate(dload_train)):
    x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device).squeeze().long()
    labels = torch.ones(x_p_d.shape[0]).long().cuda()
    scores, logits = f(x_p_d, None)
    post = nn.Softmax(dim=1)(logits)
    with torch.no_grad():
        gts.extend(y_p_d.cpu().numpy())
        features.extend(scores.cpu().numpy())
        posts.extend(post.cpu().numpy())
# Un Labeled data
l_gts, l_features, l_posts = [], [], []
for _, (l_x, l_y) in tqdm(enumerate(dload_unlabeled)):
    l_x, l_y = l_x.to(device), l_y.to(device).squeeze().long()
    # labels = torch.ones(x_p_d.shape[0]).long().cuda()
    l_scores, l_logits = f(l_x, None)
    l_post = nn.Softmax(dim=1)(l_logits)
    with torch.no_grad():
        l_gts.extend(l_y.cpu().numpy())
        l_features.extend(l_scores.cpu().numpy())
        l_posts.extend(l_post.cpu().numpy())
features = features+l_features
gts = gts + l_gts
posts = posts + l_posts

with torch.no_grad():
    features = np.array(features).reshape((-1, 128))
    gts = np.array(gts)
    posts = np.array(posts)

# Choose vis_tools
if args.vis_tools == 'tsne':
    decomp = TSNE(n_components=2)
else:
    decomp = PCA(n_components=2)
figname=f'{args.save_directory}/vis-{args.vis_tools}'
X_decomp = decomp.fit_transform(features)
cmap = get_cmap('tab10')

# Simple t-SNE(PCA)
for i in range(n_classes):
    # marker = "$" + str(i) + "$"
    marker = '.'
    indices = (gts == i)
    plt.scatter(X_decomp[indices, 0], X_decomp[indices,1], marker=marker, s=10, color=cmap(i), label="Class {}".format(i))
    plt.scatter(X_decomp[:10, 0], X_decomp[:10,1], marker=marker, s=10, color=cmap(2), label="Class {}".format(i))
plt.savefig(figname+'.png')
plt.savefig(figname+'.eps')
plt.close()

# Posterior plot
plt.scatter(X_decomp[:, 0], X_decomp[:,1], marker=marker, s=5, c=posts[:,0], cmap='seismic', vmin=0.0, vmax=1.0)
plt.savefig(figname+'post'+'.png')
plt.savefig(figname+'post'+'.eps')
plt.close()

# plt.scatter(x_p_d.cpu()[:, 0], x_p_d.cpu()[:,1], marker=marker, s=5, c=posts[:,0], cmap='seismic', vmin=0.0, vmax=1.0)
# plt.savefig(figname+'input_dim'+'.png')
# plt.savefig(figname+'input_dim'+'.eps')
# plt.close()

# import scipy.stats as st
# SeqClor = ["Blues","Oranges","Greens","Reds","Purples","YlOrBr", "RdPu", "Greys"]
# Clorlis = ['#87ceeb', '#ff8c00', '#008000', '#ff4500', '#9400d3', '#8b0000', '#ff69b4', '#808080','#00bfff','#ffefd5','#00fa9a', '#d3d3d3']
# xmin, xmax = min(X_decomp[:,0])-20, max(X_decomp[:,0])+20
# ymin, ymax = min(X_decomp[:,1])-20, max(X_decomp[:,1])+20
# # Peform the kernel density estimate
# plt.axes().set_aspect('equal')
# # plt.xlim(xmin, xmax)
# # plt.ylim(ymin, ymax)
# plt.xticks([])
# plt.yticks([])
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.array([xx.ravel(), yy.ravel()])
# for k in range(n_classes):
#     indices = (gts == k)
#     values = np.array([X_decomp[indices, 0], X_decomp[indices, 1]])
#     kernel = st.gaussian_kde(values)
#     f = np.reshape(kernel(positions).T, xx.shape)
#     # Contourf plot
#     # plt.contourf(xx, yy, f, cmap=SeqClor[i])
#     # plt.imshow(np.rot90(f), cmap=SeqClor[i], extent=[xmin, xmax, ymin, ymax])
#     plt.contour(xx, yy, f, colors=Clorlis[k])
#     # ax.clabel(cset, inline=1, fontsize=10)
# plt.savefig(figname+'_ct.png')
# plt.close()