'''
semi-supervised class conditional model
2023/12/01
'''
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
    def __init__(self, num_in, num_out, num_classes, data_classes):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        # Noise scale
        self.embed_scale = nn.Embedding(num_classes, num_out)
        self.embed_scale.weight.data.uniform_()
        # Class embed
        self.embed_class = nn.Embedding(data_classes, num_out)
        self.embed_class.weight.data.uniform_()

    def forward(self, x, s, c):
        out = self.lin(x)
        # when joint learning, to clf input data w/o noise
        if s is not None:
            gamma = self.embed_scale(s)
            out = gamma.view(-1, self.num_out) * out
        # when labeled data
        if c is not None:
            c_emb = self.embed_class(c)
            out = c_emb.view(-1, self.num_out) * out
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
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes, data_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes, data_classes)
        self.lin3 = nn.Linear(128, input_dim)
        self.lin4 = nn.Linear(128, data_classes)
    
    def forward(self, x, s, c):
        x = F.softplus(self.lin1(x, s, c))
        x = F.softplus(self.lin2(x, s, c))
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
def anneal_dsm_and_logits(scorenet, samples, sigmas, classes, anneal_power=2.0, noise_clf=False):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    if noise_clf:
        scores, _ = scorenet(perturbed_samples, labels, classes)
        _, logits = scorenet(perturbed_samples, labels, None)
    else:
        scores, _ = scorenet(perturbed_samples, labels, classes)
        _, logits = scorenet(samples, None, None)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0), logits, perturbed_samples, scores

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
    pass

# Simple Langevin
@torch.no_grad()
def sample_langevin_jl(model, x, c, n_steps=20, eps=0.5, decay=.9, temperature=0.5):
    x_sequence = [x.unsqueeze(0)]
    for s in range(n_steps):
        labels = torch.ones(x.shape[0], device=x.device).long()
        z_t = torch.rand(x.size(), device=x.device)
        scores, _ = model(x, labels, c)
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
        plt.scatter(*datas, c=posterior[:,0], cmap='seismic', s=40, vmin=0.0, vmax=1.0, alpha=0.3)
        # else:
        #     conf, pred = posterior.max(axis=1)
        #     conf = conf - 0.2
        #     plt.scatter(*datas, c=pred, cmap='tab10', s=30, alpha=conf)
    else:
        plt.scatter(*datas, alpha=0.2, color='red', s=40)
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
            dsm_losses, ce_losses, total_losses, train_accs, consistencies = [], [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                score.train()
                step += 1
                L = 0
                X = X.to(self.config.device)
                # y = y.to(self.config.device)

                # labeled data
                labeled_X, labeled_y = train_labeled_loader.__next__()
                labeled_X, labeled_y = labeled_X.to(self.config.device), labeled_y.to(self.config.device)
                
                # class conditional score
                dsm_loss_cnd, logits, l_perterbed, cond_scores = anneal_dsm_and_logits(score, labeled_X, sigmas, labeled_y, anneal_power=2.0, noise_clf=True)
                L += dsm_loss_cnd*1.0
                # class unconditional score
                dsm_loss_uncond, _, _, uncond_scores = anneal_dsm_and_logits(score, X, sigmas, None, anneal_power=2.0, noise_clf=True)
                L += dsm_loss_uncond*1.0
                # classification loss
                acc = (logits.max(1)[1] == labeled_y).float().mean()
                ce_loss = nn.CrossEntropyLoss()(logits, labeled_y)
                if ce_loss.item() > 1.0:
                    trai_post = nn.Softmax(dim=1)(logits).detach().cpu()
                    test_plot(l_perterbed.T.cpu(), os.path.join(self.args.test_path, f"train_div_{epoch}.png"), plot_lim=xy_lim, posterior=trai_post)
                L += ce_loss*0.5
                # LOSS consistency gradP(x|c) - gradP(y|x) = gradP(x)
                if False:
                    with torch.enable_grad():
                        x_in = labeled_X.detach().requires_grad_(True)
                        _, in_logits, _, _ = anneal_dsm_and_logits(score, x_in, sigmas, labeled_y, anneal_power=2.0, noise_clf=True)
                        log_prob = F.log_softmax(in_logits, dim=-1)
                        grad_prob = torch.autograd.grad(log_prob.sum(), x_in)[0]
                        # consistency_loss = nn.MSELoss()((cond_scores-grad_prob), uncond_scores[0:10])
                        # consistency_loss = nn.MSELoss()((grad_prob), uncond_scores[0:10])
                        consistency_loss = (((cond_scores.detach()-grad_prob) - uncond_scores.detach()[0:10]) ** 2).sum(dim=-1).mean(dim=0)
                    L += consistency_loss
                _, _, _, unlabel_cond_scores = anneal_dsm_and_logits(score, X[0:10], sigmas, labeled_y, anneal_power=2.0, noise_clf=True)
                if True:
                    with torch.enable_grad():
                        x_in = X[0:10].detach().requires_grad_(True)
                        _, in_logits, _, _ = anneal_dsm_and_logits(score, x_in, sigmas, None, anneal_power=2.0, noise_clf=True)
                        log_prob = F.log_softmax(in_logits, dim=-1)
                        grad_prob = torch.autograd.grad(log_prob.sum(), x_in, retain_graph=True)[0].requires_grad_(True)
                        # consistency_loss = nn.MSELoss()((cond_scores-grad_prob), uncond_scores[0:10])
                        # consistency_loss = nn.MSELoss()((grad_prob), uncond_scores[0:10])
                        consistency_loss = (((unlabel_cond_scores.detach()-grad_prob*0.1) - uncond_scores.detach()[0:10]) ** 2).sum(dim=-1).mean(dim=0)
                    L += consistency_loss*2.0
                dsm_losses.append(dsm_loss_cnd.item()+dsm_loss_uncond.item())
                ce_losses.append(ce_loss.item())
                total_losses.append(L.item())
                train_accs.append(acc.item())
                consistencies.append(consistency_loss.item())
                # backwardできていないということがわかった．
                optimizer.zero_grad()
                if epoch < 1000:
                    L.backward()
                else:
                    consistency_loss.backward()
                optimizer.step()
            
            tb_logger.add_scalar('train/dsm_loss', np.mean(dsm_losses), global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', np.mean(ce_losses), global_step=epoch)
            tb_logger.add_scalar('train/total_loss', np.mean(total_losses), global_step=epoch)
            tb_logger.add_scalar('train/acc', np.mean(train_accs), global_step=epoch)
            logging.info("epoch: {}, step: {}, dsm: {}, ce: {}, acc: {}, consistency: {}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs), np.mean(consistencies)))

            if epoch % 100 == 0:
                test_score = score
                test_score.eval()
                test_dsms, test_ces, test_acces, test_eces = [], [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    test_labels = torch.randint(3, len(sigmas), (test_X.shape[0],)).cuda()
                    with torch.no_grad():
                        if self.config.training.mode == 'joint_learning':
                            test_dsm_loss, _, t_perturbed_data, _ = anneal_dsm_and_logits(test_score, test_X, sigmas, None,
                                                            self.config.training.anneal_power,
                                                            self.config.training.noise_clf)
                            test_dsms.append(test_dsm_loss.item())
                        # test data without noise (no mean w/noise?)
                        # encoderが学習しているのはnoisy data
                        _, t_logits = test_score(test_X, test_labels, None)
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
                    scores, _ = score(xx, xx_labels, None)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    
                    # 全データに対するposteriorがほしい
                    all_data = torch.concat((X, labeled_X)).clone().cuda() #今はこうだけど，全サンプルでスコア推定すれば良いような
                    all_labels = torch.randint(0, len(sigmas), (all_data.shape[0],)).cuda()
                    _, all_logits = score(all_data, None, None)
                    all_post = nn.Softmax(dim=1)(all_logits).detach().cpu()
                    # Perform the plots
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    # Posterior score
                    xx_in = xx.detach().requires_grad_(True)
                    _, s_logits = score(xx_in, None, None)
                    log_prob = F.log_softmax(s_logits, dim=-1)
                    grad_prob = torch.autograd.grad(log_prob.sum(), xx_in)[0]
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), grad_prob.T.cpu(),
                        os.path.join(self.args.plt_fifure_path, f"posterior_scores_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    # Class 0
                    scores, _ = score(xx, xx_labels, torch.zeros(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c0 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"0_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # Class 1
                    scores, _ = score(xx, xx_labels, torch.ones(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c1 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"1_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # ベクトルの足し算したものを表示する
                    hikizan_c0 = grad_prob.cpu()*0.1-scores_log1p_c0
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"0_posterior_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                                        # ベクトルの足し算したものを表示する
                    hikizan_c1 = grad_prob.cpu()*0.1-scores_log1p_c1
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c1.T,
                                os.path.join(self.args.plt_fifure_path, f"1_posterior_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # Sampling
                    if False:
                        test_score = score.eval()
                        init_samples = torch.Tensor([[xy_lim, xy_lim], [xy_lim, -xy_lim],
                                                        [-xy_lim, xy_lim], [-xy_lim, -xy_lim]]).to(self.config.device)
                        if True:
                            uncond_samples = sample_langevin_jl(test_score, init_samples, None).detach().cpu()
                            cond_samples_0 = sample_langevin_jl(test_score, init_samples, torch.zeros(len(init_samples), dtype=int, device=self.config.device)).detach().cpu()
                            cond_samples_1 = sample_langevin_jl(test_score, init_samples, torch.ones(len(init_samples), dtype=int, device=self.config.device)).detach().cpu()
                        else:
                            samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                self.config.sampling.n_steps_each,
                                                                self.config.sampling.step_lr,
                                                                final_only=False, verbose=True,
                                                                denoise=self.config.sampling.denoise)
                        plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"uncond_samples_scoreslog_{epoch}.png"),
                                samples=uncond_samples, plot_lim=xy_lim)
                        plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c0.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"cond_0_samples_scoreslog_{epoch}.png"),
                                samples=cond_samples_0, plot_lim=xy_lim)
                        plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c1.T.cpu(),
                                os.path.join(self.args.log_sample_path, f"cond_1_samples_scoreslog_{epoch}.png"),
                                samples=cond_samples_1, plot_lim=xy_lim)
                    # del scores, scores_norm, scores_log1p#, all_samples
                    # del test_score