import numpy as np
import glob
import tqdm
import torch.nn.functional as F
import logging
import torch
import os
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt


__all__ = ['NCSNRunner']

# Model
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(num_classes, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        # when joint learning, to clf input data w/o noise
        if y is None:
            return out
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin3 = nn.Linear(128, input_dim)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)
    
class JointLearningModel(nn.Module):
    def __init__(self, input_dim, num_classes, data_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin3 = nn.Linear(128, input_dim)
        self.lin4 = nn.Linear(128, data_classes)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x), self.lin4(x)

class ClassifierModel(nn.Module):
    def __init__(self, input_dim, num_classes, data_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        # self.lin3 = nn.Linear(128, input_dim)
        self.lin4 = nn.Linear(128, data_classes)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        # return self.lin3(x), self.lin4(x)
        return 0, self.lin4(x)

# Choose model
def get_model(config):
    if config.training.mode == 'joint_learning':
        model = JointLearningModel(config.input_dim,config.model.num_classes, config.data.d_classes)
        return model.to(config.device)
    elif config.training.mode == 'clf':
        model = ClassifierModel(config.input_dim,config.model.num_classes, 2)
        return model.to(config.device)
    else:
        model = ConditionalModel(config.input_dim,config.model.num_classes)
        return model.to(config.device)

# Choose Dataset
def get_dataset(config):
    if config.data.dim == 2:
        train_data = np.load(f"{config.data.dir}/train_data.npy")
        train_labels = np.load(f"{config.data.dir}/train_labels.npy")
        test_data = np.load(f"{config.data.dir}/test_data.npy")
        test_labels = np.load(f"{config.data.dir}/test_labels.npy")
    elif config.data.dim == 1:
        train_data = np.load(f"./dataset/unidimensionally_Gaussian/sigma.5/mu_{config.data.mu}/0/train_data.npy")
        train_labels = np.load(f"./dataset/unidimensionally_Gaussian/sigma.5/mu_{config.data.mu}/0/train_labels.npy")
        test_data = np.load(f"./dataset/unidimensionally_Gaussian/sigma.5/mu_{config.data.mu}/0/test_data.npy")
        test_labels = np.load(f"./dataset/unidimensionally_Gaussian/sigma.5/mu_{config.data.mu}/0/test_labels.npy")
    # Data loaders
    labeled_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    dload_train = DataLoader(labeled_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_test = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    return dload_train, dload_test

def get_ssl_dataset(config):
    def cycle(loader):
       while True:
            for data in loader:
                yield data
    labeled_data = np.load(f"{config.data.dir}/labeled_data.npy")
    obs_labels = np.load(f"{config.data.dir}/obs_labels.npy")
    unlabeled_data = np.load(f"{config.data.dir}/unlabeled_data.npy")
    unobs_labels = np.load(f"{config.data.dir}/unobs_labels.npy")
    test_data = np.load(f"{config.data.dir}/test_data.npy")
    test_labels = np.load(f"{config.data.dir}/test_labels.npy")
    logging.info(f'Train: labeled samples: {len(labeled_data)}, unlabeled samples: {len(unlabeled_data)}')
    logging.info(f'Test : labeled samples: {len(test_data)}')
    # Data loaders
    labeled_dataset = TensorDataset(torch.from_numpy(labeled_data).float(), torch.from_numpy(obs_labels).long())
    unlabeled_dataset = TensorDataset(torch.from_numpy(unlabeled_data).float(), torch.from_numpy(unobs_labels).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    dload_train_labeled = DataLoader(labeled_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_train = DataLoader(unlabeled_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_test = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    return dload_train, dload_train_labeled, dload_test

# LOSS FUNCTION
def anneal_dsm(scorenet, samples, sigmas, anneal_power=2.0):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

def anneal_dsm_and_logits(scorenet, samples, sigmas, anneal_power=2.0, noise_clf=False):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    if noise_clf:
        scores, logits = scorenet(perturbed_samples, labels)
    else:
        scores, _ = scorenet(perturbed_samples, labels)
        _, logits = scorenet(samples, None)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0), logits, perturbed_samples

# SIGMAS
def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = [x_mod.unsqueeze(0).to('cpu')]

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                # noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                # image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                # snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                # grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.unsqueeze(0).to('cpu'))
                # if verbose:
                #     print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                #         c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.unsqueeze(0).to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return torch.cat(images)

# Simple Langevin
@torch.no_grad()
def sample_langevin(model, x, n_steps=20, eps=0.5, decay=.9, temperature=0.5):
    x_sequence = [x.unsqueeze(0)]
    for s in range(n_steps):
        labels = torch.ones(x.shape[0], device=x.device).long()
        z_t = torch.rand(x.size(), device=x.device)
        x = x + (eps / 2) * model(x, labels) + (np.sqrt(eps) * temperature * z_t)
        x_sequence.append(x.unsqueeze(0))
        eps *= decay
    return torch.cat(x_sequence)

@torch.no_grad()
def sample_langevin_jl(model, x, n_steps=20, eps=0.5, decay=.9, temperature=0.5):
    x_sequence = [x.unsqueeze(0)]
    for s in range(n_steps):
        labels = torch.ones(x.shape[0], device=x.device).long()
        z_t = torch.rand(x.size(), device=x.device)
        scores, _ = model(x, labels)
        x = x + (eps / 2) * scores + (np.sqrt(eps) * temperature * z_t)
        x_sequence.append(x.unsqueeze(0))
        eps *= decay
    return torch.cat(x_sequence)

# PLOT FIGURE
def plot_scores(datas, meshs, scores, dir, samples=None, plot_lim=None, posterior=None):
    if datas.shape[0] == 1:
        # y points
        zeros = torch.zeros(10000)
        y_cor = torch.linspace(-2, 2, 50)
        # 1dim to 2dim (x, 0)
        datas = torch.cat([datas, zeros[:datas.shape[-1]].reshape(1, -1)], dim=0)
        meshs = torch.cat([meshs, y_cor[:meshs.shape[-1]].reshape(1, -1)], dim=0)
        scores = torch.cat([scores, zeros[:scores.shape[-1]].reshape(1, -1)], dim=0)
    plt.figure(figsize=(16,12))
    # plot dataset points
    if posterior != None:
        # if posterior.size(1) == 2:
        plt.scatter(*datas, c=posterior[:,0], cmap='seismic', s=40, vmin=0.0, vmax=1.0)
        # else:
        #     conf, pred = posterior.max(axis=1)
        #     conf = conf - 0.2
        #     plt.scatter(*datas, c=pred, cmap='tab10', s=30, alpha=conf)
    else:
        plt.scatter(*datas, alpha=0.3, color='red', s=40)
    # initial point and vector direction
    plt.quiver(*meshs, *scores, width=0.002, color="0.3")
    if plot_lim != None:
        plt.xlim(-plot_lim, plot_lim)
        plt.ylim(-plot_lim, plot_lim)
    plt.plot()
    if samples is not None:
        for i in range(4): # num of samples
            plt.scatter(samples[:, i, 0], samples[:, i, 1], color=plt.get_cmap("tab20")(i), edgecolor='white', s=150)
            deltas = (samples[:, i][1:] - samples[:, i][:-1])
            deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
            for j, arrow in enumerate(deltas):
                plt.arrow(samples[j, i, 0], samples[j, i, 1], arrow[0], arrow[1], width=1e-5, head_width=2e-3, color=plt.get_cmap("tab20")(i), linewidth=3)
    plt.savefig(dir)
    plt.close()

def test_plot(datas, dir, plot_lim=None, posterior=None):
    if datas.shape[0] == 1:
        # 1dim to 2dim (x, 0)
        zeros = torch.zeros(10000)
        datas = torch.cat([datas, zeros[:datas.shape[-1]].reshape(1, -1)], dim=0)
    plt.figure(figsize=(16,12))
    # plot dataset points
    if posterior != None:
        # if posterior.size(1) == 2:
        plt.scatter(*datas, c=posterior[:,0], cmap='seismic', s=40, vmin=0.0, vmax=1.0)
        # else:
        #     conf, pred = posterior.max(axis=1)
        #     conf = conf - 0.2
        #     plt.scatter(*datas, c=pred, cmap='tab10', s=30, alpha=conf)
    else:
        plt.scatter(*datas, alpha=0.3, color='red', s=40)
    if plot_lim != None:
        plt.xlim(-plot_lim, plot_lim)
        plt.ylim(-plot_lim, plot_lim)
    plt.plot()
    plt.savefig(dir)
    plt.close()

# Checkpoints save model
def checkpoint(model, sigmas, optimizer, epoch, step, logdir, tag, device):
    model.cpu()
    ckpt_dict = {
        "model_state_dict": model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        "sigmas": sigmas,
        "epoch": epoch,
        "step": step
    }
    torch.save(ckpt_dict, os.path.join(logdir, tag))
    model.to(device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        args.plt_fifure_path = os.path.join(args.log_path, 'plots')
        args.ckpt_model_path = os.path.join(args.log_path, 'ckpts')
        args.test_path = os.path.join(args.log_path, 'test')
        os.makedirs(args.log_sample_path, exist_ok=True)
        os.makedirs(args.plt_fifure_path, exist_ok=True)
        os.makedirs(args.ckpt_model_path, exist_ok=True)
        os.makedirs(args.test_path, exist_ok=True)

    def train(self):
        train_loader, test_loader = get_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # score = torch.nn.DataParallel(score)
        optimizer = optim.Adam(score.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels
        sigmas = get_sigmas(self.config)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            losses = []
            for i, (X, y) in enumerate(train_loader):
                score.train()
                step += 1

                X = X.to(self.config.device)

                loss = anneal_dsm(score, X, sigmas, self.config.training.anneal_power)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tb_logger.add_scalar('loss', np.mean(losses), global_step=epoch)
            logging.info("epoch: {}, step: {}, loss: {}".format(epoch, step, np.mean(losses)))

            if epoch % 100 == 0:
                test_score = score
                test_score.eval()
                test_losses = []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm(test_score, test_X, sigmas, self.config.training.anneal_power)
                        test_losses.append(test_dsm_loss.item())
                tb_logger.add_scalar('test_loss', np.mean(test_losses), global_step=epoch)
                logging.info("epoch: {}, step: {}, test_loss: {}".format(epoch, step, np.mean(test_losses)))
                del test_score, test_losses

            if self.config.input_dim == 1:
                # unidimention
                if epoch % 100 == 0:
                    x_1dim = np.linspace(-xy_lim, xy_lim, 50).reshape([-1, 1])
                    x_1dim = torch.from_numpy(x_1dim.astype(np.float32)).clone().cuda()
                    labels = torch.randint(0, len(sigmas), (x_1dim.shape[0],)).cuda()
                    scores = score(x_1dim, labels).detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(X.T.cpu(), x_1dim.T.cpu(), scores.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scores_{epoch}.png"), plot_lim=xy_lim)
                    plot_scores(X.T.cpu(), x_1dim.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim)
            else:
                # 2 dim
                if epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()
                    scores = score(xx, labels).detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    # Perform the plots
                    plot_scores(X.T.cpu(), xx.T.cpu(), scores.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scores_{epoch}.png"), plot_lim=xy_lim)
                    plot_scores(X.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim)
                    # Sampling
                    if True:
                        test_score = score.eval()
                        init_samples = torch.Tensor([[xy_lim, xy_lim], [xy_lim, -xy_lim],
                                                     [-xy_lim, xy_lim], [-xy_lim, -xy_lim]]).to(self.config.device)
                        if True:
                            samples = sample_langevin(test_score, init_samples).detach().cpu()
                        else:
                            samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                self.config.sampling.n_steps_each,
                                                                self.config.sampling.step_lr,
                                                                final_only=False, verbose=True,
                                                                denoise=self.config.sampling.denoise)
                        plot_scores(X.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"samples_scoreslog_{epoch}.png"),
                                samples=samples, plot_lim=xy_lim)
                    del scores, scores_norm, scores_log1p#, all_samples
                    del test_score

    def joint_learning_train(self):
        train_loader, test_loader = get_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # score = torch.nn.DataParallel(score)
        optimizer = optim.Adam(score.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels
        sigmas = get_sigmas(self.config)
        best_test_acc = 0.0
        best_test_dsm = 2.
        for epoch in range(start_epoch, self.config.training.n_epochs):
            dsm_losses, ce_losses, total_losses, train_accs = [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                y = y.to(self.config.device)

                dsm_loss, logits, perturbed_data = anneal_dsm_and_logits(score, X, sigmas,
                                                          self.config.training.anneal_power,
                                                          self.config.training.noise_clf)
                post = nn.Softmax(dim=1)(logits).detach().cpu()
                acc = (logits.max(1)[1] == y).float().mean()
                ce_loss = nn.CrossEntropyLoss()(logits, y)

                L = dsm_loss + ce_loss
                dsm_losses.append(dsm_loss.item())
                ce_losses.append(ce_loss.item())
                total_losses.append(L.item())
                train_accs.append(acc.item())

                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            
            tb_logger.add_scalar('train/dsm_loss', np.mean(dsm_losses), global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', np.mean(ce_losses), global_step=epoch)
            tb_logger.add_scalar('train/total_loss', np.mean(total_losses), global_step=epoch)
            tb_logger.add_scalar('train/acc', np.mean(train_accs), global_step=epoch)
            logging.info("epoch: {}, step: {}, dsm: {}, ce: {}, acc: {}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs)))

            if epoch % 100 == 0:
                test_score = score
                test_score.eval()
                test_dsms, test_ces, test_acces = [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    with torch.no_grad():
                        test_dsm_loss, _, t_perturbed_data = anneal_dsm_and_logits(test_score, test_X, sigmas,
                                                          self.config.training.anneal_power,
                                                          self.config.training.noise_clf)
                        # test data without noise (no mean w/noise?)
                        _, t_logits = test_score(test_X, None)
                        test_post = nn.Softmax(dim=1)(t_logits).detach().cpu()
                        test_acc = (t_logits.max(1)[1] == test_y).float().mean()
                        test_ce_loss = nn.CrossEntropyLoss()(t_logits, test_y)
                        test_dsms.append(test_dsm_loss.item())
                        test_ces.append(test_ce_loss.item())
                        test_acces.append(test_acc.item())
                test_plot(test_X.T.cpu(), os.path.join(self.args.test_path, f"test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                # test_plot(t_perturbed_data.T.cpu(), os.path.join(self.args.test_path, f"per_test_{epoch}.png"), plot_lim=xy_lim)
                tb_logger.add_scalar('test/dsm_loss', np.mean(test_dsms), global_step=epoch)
                tb_logger.add_scalar('test/ce_loss', np.mean(test_ces), global_step=epoch)
                tb_logger.add_scalar('test/acc', np.mean(test_acces), global_step=epoch)
                logging.info("epoch: {}, step: {}, test_dsm: {}, test_ce: {}, test_acc: {}"
                             .format(epoch, step, np.mean(test_dsms), np.mean(test_ces), np.mean(test_acces)))
                # Save Best models
                if np.mean(test_acces) > best_test_acc:
                    best_test_acc = np.mean(test_acces)
                    checkpoint(score, sigmas, optimizer, epoch, step,
                                self.args.log_path, 'best_acc_model.pth', self.config.device)
                    logging.info(f"Best Test ACC: {best_test_acc}")
                if np.mean(test_dsms) < best_test_dsm:
                    best_test_dsm = np.mean(test_dsms)
                    checkpoint(score, sigmas, optimizer, epoch, step,
                                self.args.log_path, 'best_dsm_model.pth', self.config.device)
                    logging.info(f"Best Test DSM: {best_test_dsm}")
            
                del test_score, test_dsms, test_ces, test_acces, test_post
            
            if self.config.input_dim == 1:
                # unidimention
                if epoch % 100 == 0:
                    x_1dim = np.linspace(-xy_lim, xy_lim, 50).reshape([-1, 1])
                    x_1dim = torch.from_numpy(x_1dim.astype(np.float32)).clone().cuda()
                    labels = torch.randint(0, len(sigmas), (x_1dim.shape[0],)).cuda()
                    scores, _ = score(x_1dim, labels)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(X.T.cpu(), x_1dim.T.cpu(), scores.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scores_{epoch}.png"), plot_lim=xy_lim, posterior=post)
                    plot_scores(X.T.cpu(), x_1dim.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=post)
            else:
            # 2 dim
                if epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()
                    scores, _ = score(xx, labels)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    # Perform the plots
                    plot_scores(X.T.cpu(), xx.T.cpu(), scores.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scores_{epoch}.png"), plot_lim=xy_lim, posterior=post)
                    plot_scores(X.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=post)
                    if self.config.training.noise_clf:
                        plot_scores(perturbed_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                    os.path.join(self.args.plt_fifure_path, f"per_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=post)
                    else:
                        plot_scores(perturbed_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"per_scoreslog_{epoch}.png"), plot_lim=xy_lim)
                    # Sampling
                    if True:
                        test_score = score.eval()
                        init_samples = torch.Tensor([[xy_lim, xy_lim], [xy_lim, -xy_lim],
                                                        [-xy_lim, xy_lim], [-xy_lim, -xy_lim]]).to(self.config.device)
                        if True:
                            samples = sample_langevin_jl(test_score, init_samples, n_steps=self.config.sampling.n_steps,
                                                         eps=self.config.sampling.eps, decay=self.config.sampling.decay,
                                                         temperature=self.config.sampling.temperature).detach().cpu()
                        else:
                            samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                self.config.sampling.n_steps_each,
                                                                self.config.sampling.step_lr,
                                                                final_only=False, verbose=True,
                                                                denoise=self.config.sampling.denoise)
                        plot_scores(X.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"samples_scoreslog_{epoch}.png"),
                                samples=samples, plot_lim=xy_lim)
                    del scores, scores_norm, scores_log1p#, all_samples
                    del test_score

    def joint_learning_train_ssl(self):
        train_loader, train_labeled_loader, test_loader = get_ssl_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # score = torch.nn.DataParallel(score)
        optimizer = optim.Adam(score.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels 0 > 1 > 2 >..
        sigmas = get_sigmas(self.config)
        best_test_acc = 0.
        best_test_dsm = 2.
        for epoch in range(start_epoch, self.config.training.n_epochs):
            dsm_losses, ce_losses, total_losses, train_accs = [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                score.train()
                step += 1
                L = 0
                X = X.to(self.config.device)
                # y = y.to(self.config.device)

                # labeled data
                labeled_X, labeled_y = train_labeled_loader.__next__()
                labeled_X, labeled_y = labeled_X.to(self.config.device), labeled_y.to(self.config.device)
                # Proposed New Method
                if True and epoch > 3500:
                    # trainデータからスタートしてサンプリング
                    with torch.no_grad():
                        # data_apex = sample_langevin_jl_train(score, X)
                        sample_datas = sample_langevin_jl(score, X, n_steps=10, eps=0.1, decay=.9, temperature=0)
                        data_apex = sample_datas[-1]
                        # このデータを予測する
                        _, apex_logits = score(data_apex, None)
                        apex_pred = torch.argmax(nn.Softmax(dim=1)(apex_logits), dim=1)
                    _, pseudo_logits = score(X, None)
                    pseudo_ce_loss = nn.CrossEntropyLoss()(pseudo_logits, apex_pred)
                    plt.scatter(data_apex[:, 0].cpu(), data_apex[:,1].cpu(), c=(nn.Softmax(dim=1)(pseudo_logits)[:,0]).detach().cpu().numpy(), cmap='seismic', s=5, vmin=0.0, vmax=1.0)
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.savefig(os.path.join(self.args.test_path, f"0_pseudo_{epoch}.png"))
                    plt.close()
                    plt.scatter(data_apex[:, 0].cpu(), data_apex[:,1].cpu(), c=apex_pred.cpu().numpy(), s=5)
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.savefig(os.path.join(self.args.test_path, f"1_pseudo_{epoch}.png"))
                    plt.close()
                    from matplotlib.cm import get_cmap
                    plt.scatter(data_apex[:, 0].cpu(), data_apex[:,1].cpu(), c=get_cmap("Set1")(y.numpy()), s=5)
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.savefig(os.path.join(self.args.test_path, f"2_pseudo_{epoch}.png"))
                    plt.close()
                    L += pseudo_ce_loss
                    
                if self.config.training.mode == 'joint_learning':
                # if False:
                    # ノイズかけるの関数内じゃないほうが良さそう
                    samples = X
                    # samples = torch.concat((X, labeled_X)).clone().cuda()
                    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
                    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
                    noise = torch.randn_like(samples) * used_sigmas
                    perturbed_samples = samples + noise
                    # dsm
                    target = - 1 / (used_sigmas ** 2) * noise
                    scores, _ = score(perturbed_samples, labels)
                    target = target.view(target.shape[0], -1)
                    scores = scores.view(scores.shape[0], -1)
                    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** self.config.training.anneal_power
                    dsm_loss = loss.mean(dim=0)
                    L += dsm_loss
                    dsm_losses.append(dsm_loss.item())
                # ラベル付きはlogitだす
                labeled_labels = torch.randint(0, len(sigmas), (labeled_X.shape[0],), device=labeled_X.device)
                l_used_sigmas = sigmas[labeled_labels].view(labeled_X.shape[0], *([1] * len(labeled_X.shape[1:])))
                l_noise = torch.randn_like(labeled_X) * l_used_sigmas
                l_perturbed_samples = labeled_X + l_noise
                _, logits = score(l_perturbed_samples, labeled_labels)
                acc = (logits.max(1)[1] == labeled_y).float().mean()
                ce_loss = nn.CrossEntropyLoss()(logits, labeled_y)
                if ce_loss.item() > 1.0:
                    trai_post = nn.Softmax(dim=1)(logits).detach().cpu()
                    test_plot(l_perturbed_samples.T.cpu(), os.path.join(self.args.test_path, f"train_{epoch}.png"), plot_lim=xy_lim, posterior=trai_post)
                # dsm_loss, logits, perturbed_data = anneal_dsm_and_logits(score, X, sigmas,
                #                                           self.config.training.anneal_power,
                #                                           self.config.training.noise_clf)

                L += ce_loss
                ce_losses.append(ce_loss.item())
                total_losses.append(L.item())
                train_accs.append(acc.item())

                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            
            tb_logger.add_scalar('train/dsm_loss', np.mean(dsm_losses), global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', np.mean(ce_losses), global_step=epoch)
            tb_logger.add_scalar('train/total_loss', np.mean(total_losses), global_step=epoch)
            tb_logger.add_scalar('train/acc', np.mean(train_accs), global_step=epoch)
            logging.info("epoch: {}, step: {}, dsm: {}, ce: {}, acc: {}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs)))

            if epoch % 100 == 0:
            # if epoch > 3500:
                test_score = score
                test_score.eval()
                test_dsms, test_ces, test_acces, test_eces = [], [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    with torch.no_grad():
                        if self.config.training.mode == 'joint_learning':
                            test_dsm_loss, _, t_perturbed_data = anneal_dsm_and_logits(test_score, test_X, sigmas,
                                                            self.config.training.anneal_power,
                                                            self.config.training.noise_clf)
                            test_dsms.append(test_dsm_loss.item())
                        # test data without noise (no mean w/noise?)
                        _, t_logits = test_score(test_X, None)
                        test_post = nn.Softmax(dim=1)(t_logits).detach().cpu()
                        test_acc = (t_logits.max(1)[1] == test_y).float().mean()
                        test_ce_loss = nn.CrossEntropyLoss()(t_logits, test_y)
                        test_ces.append(test_ce_loss.item())
                        test_acces.append(test_acc.item())
                # Save Best models
                if np.mean(test_acces) > best_test_acc:
                    best_test_acc = np.mean(test_acces)
                    checkpoint(score, sigmas, optimizer, epoch, step,
                                self.args.log_path, 'best_acc_model.pth', self.config.device)
                    logging.info(f"Best Test ACC: {best_test_acc}")
                if np.mean(test_dsms) < best_test_dsm:
                    best_test_dsm = np.mean(test_dsms)
                    checkpoint(score, sigmas, optimizer, epoch, step,
                                self.args.log_path, 'best_dsm_model.pth', self.config.device)
                    logging.info(f"Best Test DSM: {best_test_dsm}")
                checkpoint(score, sigmas, optimizer, epoch, step,
                    self.args.ckpt_model_path, f'ckpt_model_{epoch}.pth', self.config.device)
                # have some trouble
                # from netcal.metrics import ECE
                # from netcal.presentation import ReliabilityDiagram
                # calibration_score = ECE(10).measure(np.array(test_post), np.array(test_y.detach().cpu()))
                # ReliabilityDiagram(10).plot(np.array(test_post), np.array(test_y.detach().cpu())).savefig(os.path.join(self.args.test_path, f"RD_test_{epoch}.png"))
                # plt.close()
                test_plot(test_X.T.cpu(), os.path.join(self.args.test_path, f"test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                # test_plot(t_perturbed_data.T.cpu(), os.path.join(self.args.test_path, f"per_test_{epoch}.png"), plot_lim=xy_lim)
                tb_logger.add_scalar('test/dsm_loss', np.mean(test_dsms), global_step=epoch)
                tb_logger.add_scalar('test/ce_loss', np.mean(test_ces), global_step=epoch)
                tb_logger.add_scalar('test/acc', np.mean(test_acces), global_step=epoch)
                # tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                logging.info("epoch: {}, step: {}, test_dsm: {}, test_ce: {}, test_acc: {}"
                             .format(epoch, step, np.mean(test_dsms), np.mean(test_ces), np.mean(test_acces)))
                del test_score, test_dsms, test_ces, test_acces, test_post
                
                # if epoch % 100 == 0:
                if self.config.training.mode == 'joint_learning' and epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    xx_labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()
                    scores, _ = score(xx, xx_labels)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    
                    # 全データに対するposteriorがほしい
                    all_data = torch.concat((X, labeled_X)).clone().cuda() #今はこうだけど，全サンプルでスコア推定すれば良いような
                    all_labels = torch.randint(0, len(sigmas), (all_data.shape[0],)).cuda()
                    _, all_logits = score(all_data, all_labels)
                    all_post = nn.Softmax(dim=1)(all_logits).detach().cpu()
                    # Perform the plots
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    # Sampling
                    if True:
                        test_score = score.eval()
                        init_samples = torch.Tensor([[xy_lim, xy_lim], [xy_lim, -xy_lim],
                                                        [-xy_lim, xy_lim], [-xy_lim, -xy_lim]]).to(self.config.device)
                        if True:
                            samples = sample_langevin_jl(test_score, init_samples).detach().cpu()
                        else:
                            samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                self.config.sampling.n_steps_each,
                                                                self.config.sampling.step_lr,
                                                                final_only=False, verbose=True,
                                                                denoise=self.config.sampling.denoise)
                        plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"samples_scoreslog_{epoch}.png"),
                                samples=samples, plot_lim=xy_lim)
                    del scores, scores_norm, scores_log1p#, all_samples
                    del test_score
'''
    def classification_train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=False)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        # optimizer = get_optimizer(self.config, score)
        if self.config.training.fix_enc_params:
            logging.info("encoder params are fixed.")
            optimizer = optim_except_fix_params(self.config, score)
        else:
            optimizer = get_optimizer(self.config, score.parameters())
        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        sigmas = get_sigmas(self.config)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0], strict=True)
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            # optimizer.load_state_dict(states[1])
            # start_epoch = states[2]
            # step = states[3]
            # if self.config.model.ema:
            #     ema_helper.load_state_dict(states[4])

        # optimizer = optim_except_fix_params(self.config, score)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass
        
        best_acc = 0
        n_iter = len(dataloader)
        print('clssification n_iter:{}'.format(n_iter))
        for epoch in range(start_epoch, self.config.training.n_epochs):
            loss_per_epoch = .0
            acc_per_epoch = .0
            for _, (X, y) in tqdm.tqdm(enumerate(dataloader)):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                y = y.to(self.config.device)

                _, logits = score(X, y)
                loss = torch.nn.CrossEntropyLoss()(logits, y) 
                acc = (logits.max(1)[1] == y).float().mean()
                loss_per_epoch += loss.item()
                acc_per_epoch += acc.item()
                # tb_logger.add_scalar('train/loss', loss, global_step=step)
                # tb_logger.add_scalar('train/acc', acc, global_step=step)
                # tb_hook()
                # logging.info("step: {}, loss: {}, acc: {}".format(step, loss.item(), acc.item()))

                # この前後で重みを見る
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0
            # Show epoch 
            logging.info("epoch: {}, step:{}, loss: {}, acc: {}".format(epoch, step, loss_per_epoch/n_iter, acc_per_epoch/n_iter))
            tb_logger.add_scalar('train/loss', loss_per_epoch/n_iter, global_step=epoch)
            tb_logger.add_scalar('train/acc', acc_per_epoch/n_iter, global_step=epoch)
            tb_hook()
            
            
            # Test data
            if epoch % 10 == 0:
                corrects, losses, confs, gts = [], [], [], []
                
                if self.config.model.ema:
                    if self.config.training.fix_enc_params:
                        test_score = ema_helper.ema_copy_joint(score)
                    else:
                        test_score = ema_helper.ema_copy(score)
                else:
                    test_score = score

                test_score.eval()
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)
                    test_y = test_y.to(self.config.device)

                    with torch.no_grad():
                        _, test_logits = test_score(test_X, test_y)
                        test_loss = torch.nn.CrossEntropyLoss()(test_logits, test_y).cpu().numpy()
                        test_correct = (test_logits.max(1)[1] == test_y).float().cpu().numpy()
                        # test_acc = (test_logits.max(1)[1] == test_y).float().mean()

                        losses.append(test_loss.item())
                        corrects.extend(test_correct)
                        confs.extend(F.softmax(test_logits, 1).cpu().numpy()) #Posterior
                        gts.extend(test_y.cpu().numpy())
                
                del test_score
                test_loss_avg = np.mean(losses)
                test_acc_avg = np.mean(corrects)
                confs = np.array(confs).reshape((-1, 10))
                gts = np.array(gts)
                calibration_score = ECE(10).measure(confs, gts)
                tb_logger.add_scalar('test/loss', test_loss_avg, global_step=epoch)
                tb_logger.add_scalar('test/acc', test_acc_avg, global_step=epoch)
                tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                test_tb_hook()
                logging.info("step: {}, test_loss: {:.5f}, test_acc: {:.5f}, test_ece: {:.5f}".format(step, test_loss.item(), test_acc_avg.item(), calibration_score))

                # Save check point
                if test_acc_avg > best_acc:
                    logging.info("Best ACC. save check point.")
                    best_acc = test_acc_avg
                    states = [score.state_dict(), optimizer.state_dict(), epoch, step, ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_best_model.pth'))

            # Snapshot freq
            if epoch % self.config.training.snapshot_freq == 0:
                states = [score.state_dict(), optimizer.state_dict(), epoch, step,]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))

                if self.config.training.snapshot_sampling:
                    if self.config.model.ema:
                        if self.config.training.fix_enc_params:
                            test_score = ema_helper.ema_copy_joint(score)
                        else:
                            test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()

                    ## Different part from NeurIPS 2019.
                    ## Random state will be affected because of sampling during training time.
                    init_samples = torch.rand(36, self.config.data.channels,
                                            self.config.data.image_size, self.config.data.image_size,
                                            device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                    all_samples = anneal_Langevin_dynamics_joint_learning(init_samples, test_score, sigmas.cpu().numpy(),
                                                        self.config.sampling.n_steps_each,
                                                        self.config.sampling.step_lr,
                                                        final_only=True, verbose=True,
                                                        denoise=self.config.sampling.denoise)

                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, 6)
                    save_image(image_grid,
                            os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                    torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                    del test_score
                    del all_samples

    def joint_learning_train(self):
        # Make Datasubset extract index.
        from torch.utils.data import Dataset
        class DataSubset(Dataset):
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset
                inds = np.arange(len(base_dataset))
                self.inds = inds

            def __getitem__(self, index):
                base_ind = self.inds[index]
                data, target = self.base_dataset[base_ind]
                return data, target, base_ind

            def __len__(self):
                return len(self.inds)
        
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataset = DataSubset(dataset)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=False)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        optimizer = get_optimizer(self.config, score.parameters())
        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        sigmas = get_sigmas(self.config)

        hook = test_hook = None

        def tb_hook():
            pass

        def test_tb_hook():
            pass
        
        best_acc = 0.
        best_loss = 1.5
        n_iter = len(dataloader)
        print('joint learning n_iter:{}'.format(n_iter))
        for epoch in range(start_epoch, self.config.training.n_epochs):
            total_loss_per_epoch = 0.
            acc_per_epoch = 0.
            dsm_loss_per_epoch = 0.
            ce_loss_per_epoch = 0.
            noise_lebels = []
            inds = []
            ce_loss_list = []
            for _, (X, y, ind) in tqdm.tqdm(enumerate(dataloader)):
                score.train()
                step += 1
                L = 0.

                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                y = y.to(self.config.device)

                # clf onlyの場合はまた後で
                dsm_losses, logits, noise_lebel = anneal_dsm_score_estimation_jointlearning_withCE(score, X, 
                                                                         sigmas, None,
                                                                         self.config.training.anneal_power,
                                                                         hook)
                L += self.config.losses.dsm_weight * dsm_losses.mean(dim=0)

                if self.config.losses.clf_weight > 0:
                    ce_loss = torch.nn.CrossEntropyLoss()(logits, y)
                    acc = (logits.max(1)[1] == y).float().mean()
                    L += self.config.losses.clf_weight * ce_loss

                dsm_loss_per_epoch += dsm_losses.mean(dim=0).item()
                ce_loss_per_epoch += ce_loss.item()
                total_loss_per_epoch += L.item()
                acc_per_epoch += acc.item()
                noise_lebels.append(noise_lebel.cpu().view(-1).tolist())
                inds.append(ind.tolist())
                ce_loss_list.append(ce_loss.item())
                # tb_logger.add_scalar('train/loss', loss, global_step=step)
                # tb_logger.add_scalar('train/acc', acc, global_step=step)
                # tb_hook()
                # logging.info("step: {}, loss: {}, acc: {}".format(step, loss.item(), acc.item()))

                # この前後で重みを見る
                optimizer.zero_grad()
                L.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0
            # Show epoch 
            logging.info("epoch: {}, step:{}, loss: {} ({}+{}), acc: {}".format(epoch, step, total_loss_per_epoch/n_iter, ce_loss_per_epoch/n_iter, dsm_loss_per_epoch/n_iter, acc_per_epoch/n_iter))
            tb_logger.add_scalar('train/dsm_loss', dsm_loss_per_epoch/n_iter, global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', ce_loss_per_epoch/n_iter, global_step=epoch)
            tb_logger.add_scalar('train/total_loss', total_loss_per_epoch/n_iter, global_step=epoch)
            tb_logger.add_scalar('train/epoch_acc', acc_per_epoch/n_iter, global_step=epoch)
            tb_logger.add_histogram("dsm_losses", dsm_losses.cpu(), global_step=epoch)
            tb_hook()
            
            # save worst loss model
            if ((ce_loss_per_epoch/n_iter) > 1.4) and epoch > 300:
                logging.info("Small deverged LOSS. save check point.")
                states = [score.state_dict(), optimizer.state_dict(), epoch, step, ce_loss_list, noise_lebels, inds]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(self.args.log_path, f'checkpoint_divloss_model_{epoch}.pth'))
            # save best loss model
            if ((ce_loss_per_epoch/n_iter) < best_loss) and epoch > 100:
                logging.info("Worst LOSS. save check point.")
                best_loss = ce_loss_per_epoch/n_iter
                states = [score.state_dict(), optimizer.state_dict(), epoch, step, ce_loss_list, noise_lebels, inds]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(self.args.log_path, 'checkpoint_bestloss_model.pth'))            

            # Test data
            if epoch % 5 == 0:
                corrects, losses, confs, gts = [], [], [], []
                
                if self.config.model.ema:
                    if self.config.training.fix_enc_params:
                        test_score = ema_helper.ema_copy_joint(score)
                    else:
                        test_score = ema_helper.ema_copy(score)
                else:
                    test_score = score

                test_score.eval()
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)
                    test_y = test_y.to(self.config.device)

                    with torch.no_grad():
                        _, test_logits = test_score(test_X, None)
                        test_loss = torch.nn.CrossEntropyLoss()(test_logits, test_y).cpu().numpy()
                        test_correct = (test_logits.max(1)[1] == test_y).float().cpu().numpy()
                        # test_acc = (test_logits.max(1)[1] == test_y).float().mean()

                        losses.append(test_loss.item())
                        corrects.extend(test_correct)
                        confs.extend(F.softmax(test_logits, 1).cpu().numpy()) #Posterior
                        gts.extend(test_y.cpu().numpy())
                
                del test_score
                test_loss_avg = np.mean(losses)
                test_acc_avg = np.mean(corrects)
                confs = np.array(confs).reshape((-1, 10))
                gts = np.array(gts)
                calibration_score = ECE(10).measure(confs, gts)
                tb_logger.add_scalar('test/loss', test_loss_avg, global_step=epoch)
                tb_logger.add_scalar('test/acc', test_acc_avg, global_step=epoch)
                tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                test_tb_hook()
                logging.info("step: {}, test_loss: {:.5f}, test_acc: {:.5f}, test_ece: {:.5f}".format(step, test_loss.item(), test_acc_avg.item(), calibration_score))

                # Save check point
                if test_acc_avg > best_acc:
                    logging.info("Best ACC. save check point.")
                    best_acc = test_acc_avg
                    states = [score.state_dict(), optimizer.state_dict(), epoch, step, ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_best_model.pth'))

            # Snapshot freq
            if epoch % self.config.training.snapshot_freq == 0:
                states = [score.state_dict(), optimizer.state_dict(), epoch, step,]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                if epoch % (self.config.training.snapshot_freq*10) == 0:
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))

                if self.config.training.snapshot_sampling:
                    if self.config.model.ema:
                        if self.config.training.fix_enc_params:
                            test_score = ema_helper.ema_copy_joint(score)
                        else:
                            test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()

                    ## Different part from NeurIPS 2019.
                    ## Random state will be affected because of sampling during training time.
                    init_samples = torch.rand(36, self.config.data.channels,
                                            self.config.data.image_size, self.config.data.image_size,
                                            device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                    all_samples = anneal_Langevin_dynamics_joint_learning(init_samples, test_score, sigmas.cpu().numpy(),
                                                        self.config.sampling.n_steps_each,
                                                        self.config.sampling.step_lr,
                                                        final_only=True, verbose=True,
                                                        denoise=self.config.sampling.denoise)

                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, 6)
                    save_image(image_grid,
                            os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                    torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                    del test_score
                    del all_samples
'''