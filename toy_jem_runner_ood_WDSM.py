'''
supervised class conditional model
2024/1/22
'''
import numpy as np
import torch.nn.functional as F
import logging, torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.optim as optim
import matplotlib.pyplot as plt


__all__ = ['OoD_JEMRunner_WDSM']

# ----------------------------------------
# Model
# ----------------------------------------
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        # Noise scale embed
        self.embed_scale = nn.Embedding(num_classes, num_out)
        self.embed_scale.weight.data.uniform_()

    def forward(self, x, s):
        out = self.lin(x)
        gamma = self.embed_scale(s)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class jemModel(nn.Module):
    def __init__(self, input_dim, num_classes, data_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin4 = nn.Linear(128, data_classes)
    
    def forward(self, x, s):
        x = F.softplus(self.lin1(x, s))
        x = F.softplus(self.lin2(x, s))
        return self.lin4(x)

# Choose model
def get_model(config):
    # if config.training.mode == 'joint_learning':
    model = jemModel(config.input_dim, config.model.num_classes, config.data.d_classes)
    return model.to(config.device)

# ----------------------------------------
# Choose Dataset
# ----------------------------------------
def get_dataset(config):
    train_data = np.load(f"{config.data.dir}/train_data.npy")
    train_labels = np.load(f"{config.data.dir}/train_labels.npy")
    test_data = np.load(f"{config.data.dir}/test_data_ood.npy")
    test_labels = np.load(f"{config.data.dir}/test_labels_ood.npy")
    # Data loaders
    labeled_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    dload_train = DataLoader(labeled_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_test = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    return dload_train, dload_test

def cycle(loader):
    while True:
        for data in loader:
            yield data

# ----------------------------------------
# LOSS FUNCTION
# ----------------------------------------
def make_perturbed_samples(samples, sigmas):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    return perturbed_samples, used_sigmas, noise, labels

def anneal_jem_dsm(scorenet, samples, sigmas, classes=None, anneal_power=2.0):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    perturbed_samples = perturbed_samples.detach().requires_grad_(True)
    target = - 1 / (used_sigmas ** 2) * noise
    logits = scorenet(perturbed_samples, labels)
    if classes == None:
        # Unconditional score
        scores = torch.autograd.grad(logits.logsumexp(1).sum(), perturbed_samples, create_graph=True)[0]
    else:
        # Conditional score
        scores = torch.autograd.grad(torch.gather(logits, 1, classes[:, None]).sum(), perturbed_samples, create_graph=True)[0]
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0), perturbed_samples, scores

def anneal_ce_loss(model, samples, sigmas, classes):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    # conditionつけてlogitだしてる
    logits = model(perturbed_samples, labels)
    acc = (logits.max(1)[1] == classes).float().mean()
    ce_loss = nn.CrossEntropyLoss()(logits, classes)
    return ce_loss, perturbed_samples, acc

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

# Simple Langevin for EBMs
# w/o aneel
def sample_langevin_ebm(model, x, c, n_steps=20, eps=0.5, decay=.9, temperature=0.5):
    pass

# Simple Langevin for EBMs
def anneal_langevin_ebm(model, x, conds, sigmas, n_steps_each=5, step_lr=0.00005, final_only=False, denoise=True, verbose=True):
    print('---Aneeal Sampling---')
    x_sequence = [x.unsqueeze(0).to('cpu')]
    # with torch.no_grad():
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x.shape[0], device=x.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            x = x.detach().requires_grad_(True)
            logits = model(x, labels)
            cs = torch.full((logits.shape[0],1), fill_value=conds, dtype=int).cuda()
            grad = torch.autograd.grad(torch.gather(logits, 1, cs).sum(), x, create_graph=True)[0]
            noise = torch.randn_like(x)
            x = x + step_size * grad + noise * np.sqrt(step_size.cpu() * 2)
            if not final_only:
                x_sequence.append(x.unsqueeze(0).detach().to('cpu'))
            if verbose:
                noise = noise.reshape(-1, *noise.shape[2:])
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x.view(x.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size.cpu() / 2.) * grad_norm / noise_norm
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))
    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x.shape[0], device=x.device)
        last_noise = last_noise.long()
        logits = model(x, last_noise)
        cs = torch.full((logits.shape[0],1), fill_value=conds, dtype=int).cuda()
        grad = torch.autograd.grad(torch.gather(logits, 1, cs).sum(), x, create_graph=True)[0]
        x = x + sigmas[-1] ** 2 * grad
        # x = x - grad
        x_sequence.append(x.unsqueeze(0).detach().to('cpu'))
    if final_only:
        return [x.to('cpu')]
    else:
        return torch.cat(x_sequence)

# PLOT FIGURE
def plot_scores(datas, meshs, scores, dir, samples=None, plot_lim=None, posterior=None):
    plt.figure(figsize=(16,12))
    # plot dataset points
    if posterior != None:
        plt.scatter(*datas, c=posterior[:,0], cmap='seismic', s=40, vmin=0.0, vmax=1.0, alpha=0.2)
    else:
        plt.scatter(*datas, alpha=0.2, color='red', s=40)
    # initial point and vector direction
    plt.quiver(*meshs, *scores, width=0.002, color="0.1")
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
    if datas.shape[0] == 1: # 1dim to 2dim (x, 0)
        zeros = torch.zeros(10000)
        datas = torch.cat([datas, zeros[:datas.shape[-1]].reshape(1, -1)], dim=0)
    plt.figure(figsize=(16,12))
    # plot dataset points
    if posterior != None:
        plt.scatter(*datas, c=posterior[:,0], cmap='seismic', s=40, vmin=0.0, vmax=1.0)
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

def make_any_scores(model, xx, xx_labels, condition=None):
    xx = xx.detach().requires_grad_(True)
    xx_logits = model(xx, xx_labels)
    if condition == None:
        scores = torch.autograd.grad(xx_logits.logsumexp(1).sum(), xx, create_graph=True)[0]
    else:
        condition = torch.full((xx.shape[0],1), fill_value=condition, dtype=int).cuda()
        scores = torch.autograd.grad(torch.gather(xx_logits, 1, condition).sum(), xx, create_graph=True)[0]
    scores = scores.detach().cpu()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    return scores_log1p.detach().cpu()

def compare_dist(dset, xy_data, post):
    from scipy.stats import gaussian_kde
    from scipy.integrate import dblquad
    # KLダイバージェンスを計算する関数
    def kl_divergence(kde_p, kde_q, xmin, xmax, ymin, ymax):
        # 積分領域内の各点でのKLダイバージェンスの値を計算
        integrand = lambda x, y: kde_p([x, y]) * np.log(kde_p([x, y]) / kde_q([x, y]))
        return dblquad(integrand, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]
    
    xy_data = xy_data[post[:,0] > 0.5]

    # カーネル密度推定によるPDFの推定
    kde_X = gaussian_kde(dset.T.cpu())
    # kde_Y = gaussian_kde(dset.T.cpu())
    kde_Y = gaussian_kde(xy_data.T.cpu())

    # KLダイバージェンスの計算（積分領域を適宜設定）
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    kl_div = kl_divergence(kde_X, kde_Y, xmin, xmax, ymin, ymax)
    return kl_div

class OoD_JEMRunner_WDSM():
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
        # plots folders
        folders = ['uncond_scores', 'cond_scores/in-dist', 'cond_scores/outof-dist', 
                   'sampling', 'samples', 'distribution', 'distribution']
        for folder in folders:
            path = os.path.join(args.plt_fifure_path, folder)
            os.makedirs(path, exist_ok=True)
        for i in range(self.config.model.num_classes):
            path = os.path.join(args.plt_fifure_path, 'distribution', f'sigma{i}')
            os.makedirs(path, exist_ok=True)


    def train_sl(self):
        train_loader, test_loader = get_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu)
        tb_logger = self.config.tb_logger

        jem_model = get_model(self.config)
        optimizer = optim.Adam(jem_model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels 0 > 1 > 2 >..
        sigmas = get_sigmas(self.config)
        best_test_acc = 0.
        best_test_dsm = 2.
        best_kl_div = 100
        pseudo_dloader = None
        for epoch in range(start_epoch, self.config.training.n_epochs):
            dsm_losses, ce_losses, total_losses, train_accs, train_accs_p = [], [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                jem_model.train()
                step += 1
                L = 0
                X = X.to(self.config.device)
                y = y.to(self.config.device)

                if pseudo_dloader is not None:
                    pseudo_X, pseudo_Y = pseudo_dloader.__next__()
                    pseudo_X, pseudo_Y = pseudo_X.to(self.config.device), pseudo_Y.to(self.config.device)
                
                # classification
                ce_loss, p_samples_clf, acc = anneal_ce_loss(jem_model, X, sigmas, y)
                # pseudo_classification
                if pseudo_dloader is not None:
                    ce_loss_p, _, acc_p = anneal_ce_loss(jem_model, pseudo_X, sigmas, pseudo_Y)
                    L += ce_loss_p
                    train_accs_p.append(acc_p.item())
                    tb_logger.add_scalar('train/acc_p', np.mean(train_accs_p), global_step=epoch)
                
                # class unconditional
                dsm_uncond, p_samples_uncond, scores_uncond = anneal_jem_dsm(jem_model, X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)

                # Total LOSS
                L += dsm_uncond + ce_loss
                # LOSS SPIKE CHECK
                # if ce_loss.item() > 1.0:
                #     trai_post = nn.Softmax(dim=1)(logits).detach().cpu()
                #     test_plot(l_perterbed.T.cpu(), os.path.join(self.args.test_path, f"train_div_{epoch}.png"), plot_lim=xy_lim, posterior=trai_post)
                dsm_losses.append(dsm_uncond.item())
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
            logging.info("epoch: {}, step: {}, dsm: {}, ce: {}, acc: {}, pacc: {}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs), np.mean(train_accs_p)))
            # Eval
            if epoch % 100 == 0:
                test_jem = jem_model
                test_jem.eval()
                test_dsms, test_ces, test_acces, test_eces = [], [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    # 一番小さいnoiseがかかっている体で
                    test_labels = torch.randint(len(sigmas)-1, len(sigmas), (test_X.shape[0],)).cuda()
                    test_dsm_uncond, t_perturbed_data, _ = anneal_jem_dsm(test_jem, test_X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)
                    test_dsms.append(test_dsm_uncond.item())
                    with torch.no_grad():
                        # class unconditional
                        t_logits = test_jem(test_X, test_labels)
                        test_post = nn.Softmax(dim=1)(t_logits).detach().cpu()
                        test_acc = (t_logits.max(1)[1] == test_y).float().mean()
                        test_ce_loss = nn.CrossEntropyLoss()(t_logits, test_y)
                        test_ces.append(test_ce_loss.item())
                        test_acces.append(test_acc.item())
                test_plot(test_X.T.cpu(), os.path.join(self.args.test_path, f"test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                # test_plot(t_perturbed_data.T.cpu(), os.path.join(self.args.test_path, f"perturbed_test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                tb_logger.add_scalar('test/dsm_loss', np.mean(test_dsms), global_step=epoch)
                tb_logger.add_scalar('test/ce_loss', np.mean(test_ces), global_step=epoch)
                tb_logger.add_scalar('test/acc', np.mean(test_acces), global_step=epoch)
                # tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                logging.info("epoch: {}, step: {}, test_dsm: {}, test_ce: {}, test_acc: {}"
                             .format(epoch, step, np.mean(test_dsms), np.mean(test_ces), np.mean(test_acces)))
                del test_jem, test_dsms, test_ces, test_acces, test_post
                
                # make pseudo data
                # if epoch % 500 ==0 and epoch > 999:
                if epoch % 500 ==0:
                    test_model = jem_model.eval()
                    # ∇_x log p(x | y = in-distribution)を計算 sampling
                    init_samples = xy_lim*torch.randn(self.config.sampling.num_samples, 2).to(self.config.device)
                    if epoch != 0:
                        if self.config.sampling.mode == 'simple':
                            samples = sample_langevin_ebm(test_model, init_samples, c=0, n_steps=self.config.sampling.n_steps,
                                                        eps=self.config.sampling.eps, decay=self.config.sampling.decay, temperature=self.config.sampling.temperature).detach().cpu()
                        else:
                            samples = anneal_langevin_ebm(test_model, init_samples, 0, sigmas,
                                                    n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                                    final_only=self.config.sampling.final_only, denoise=self.config.sampling.denoise,
                                                    verbose=self.config.sampling.verbose).detach().cpu()
                        # samplingを可視化
                        plt.figure(figsize=(16,12))
                        plt.scatter(*X.T.cpu(), alpha=0.2, color='red', s=40)
                        for i in range(10): # num of samples
                            plt.scatter(samples[:, i, 0], samples[:, i, 1], color=plt.get_cmap("tab20")(i), edgecolor='white', s=150)
                            deltas = (samples[:, i][1:] - samples[:, i][:-1])
                            deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
                            for j, arrow in enumerate(deltas):
                                plt.arrow(samples[j, i, 0], samples[j, i, 1], arrow[0], arrow[1], width=1e-5, head_width=2e-3, color=plt.get_cmap("tab20")(i), linewidth=3)
                        plt.xlim(-xy_lim*2, xy_lim*2)
                        plt.ylim(-xy_lim*2, xy_lim*2)
                        plt.plot()
                        plt.savefig(os.path.join(self.args.plt_fifure_path, f"sampling/pseudo_sampleing_{epoch}.png"))
                        plt.close()
                        # 最終サンプルだけ抽出
                        final_pseudo_samples = samples[-1]
                    else:
                        final_pseudo_samples = init_samples.cpu()
                    plt.figure(figsize=(16,12))
                    plt.scatter(*final_pseudo_samples.T.cpu(), alpha=0.2, color='blue', s=40)
                    plt.scatter(*X.T.cpu(), alpha=0.2, color='red', s=40)
                    plt.xlim(-xy_lim*2, xy_lim*2)
                    plt.ylim(-xy_lim*2, xy_lim*2)
                    plt.plot()
                    plt.savefig(os.path.join(self.args.plt_fifure_path, f"samples/pseudo_sample_{epoch}.png"))
                    plt.close()
                    labels = np.ones(final_pseudo_samples.shape[0], dtype=int)
                    # datasetに追加，ここ毎回新しくなってる
                    pseudo_dataset = TensorDataset(final_pseudo_samples.float(), torch.from_numpy(labels))
                    dloader = DataLoader(pseudo_dataset, batch_size=self.config.training.batch_size, shuffle=True)
                    pseudo_dloader = cycle(dloader)
                    np.save(os.path.join(self.args.ckpt_model_path, f'final_pseudo_samples_{epoch}.npy'), final_pseudo_samples.numpy())

                # Plot
                if epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    xx_labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()

                    # P(y|x) for all samples
                    all_data = X.clone().cuda()
                    all_labels = torch.randint(0, len(sigmas), (all_data.shape[0],)).cuda()
                    all_logits = jem_model(all_data, all_labels)
                    all_post = nn.Softmax(dim=1)(all_logits).detach().cpu()

                    # Posterior Distribution
                    with torch.no_grad():
                        X, Y = np.meshgrid(np.linspace(-xy_lim, xy_lim, 100), (np.linspace(-xy_lim, xy_lim, 100)))
                        xy = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 100), np.linspace(-xy_lim, xy_lim, 100)), axis=-1).reshape(-1, 2)
                        xy = torch.from_numpy(xy.astype(np.float32)).clone().cuda()
                        for l in range (len(sigmas)):
                            # several_noise
                            xy_labels = torch.randint(l, l+1, (xy.shape[0],)).cuda()
                            logits = jem_model(xy, xy_labels).detach().cpu()
                            post = F.softmax(logits, dim=1).numpy()
                            plt.figure(figsize=(16,12))
                            plt.contourf(X, Y, post[:,0].reshape(100,100), 20, cmap='seismic')
                            # plt.colorbar(cp) # Add a colorbar to a plot
                            plt.savefig(os.path.join(self.args.plt_fifure_path, f"distribution/sigma{l}/post_{epoch}.png"))
                            plt.close()

                    if True:
                        kl_dv = compare_dist(all_data, xy, post)
                        tb_logger.add_scalar('train/kl_div', kl_dv, global_step=epoch)
                        if best_kl_div > kl_dv:
                            best_kl_div = kl_dv
                            logging.info("best kl div loss! {}".format(kl_dv))

                    uncond = make_any_scores(jem_model, xx, xx_labels, condition=None)
                    cond_0 = make_any_scores(jem_model, xx, xx_labels, condition=0)
                    cond_1 = make_any_scores(jem_model, xx, xx_labels, condition=1)
                    plot_scores(all_data.T.cpu(), xx.T.detach().cpu(), uncond.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"uncond_scores/scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    plot_scores(all_data.T.cpu(), xx.T.detach().cpu(), cond_0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/in-dist/0_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    plot_scores(all_data.T.cpu(), xx.T.detach().cpu(), cond_1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/outof-dist/1_scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
    def sampling(self):
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        # model stracture
        jem_model = get_model(self.config)
        test_model = jem_model.eval()
        # score = torch.nn.DataParallel(score)
        # Loading model weight
        epoch = 'this is change'
        # Noise labels 0 > 1 > 2 >..
        sigmas = get_sigmas(self.config)
        # ∇_x log p(x | y = in-distribution)を計算 sampling
        init_samples = xy_lim*torch.randn(self.config.sampling.num_samples, 2).to(self.config.device)
        if self.config.sampling.mode == 'simple':
            samples = sample_langevin_ebm(test_model, init_samples, c=0, n_steps=self.config.sampling.n_steps,
                                        eps=self.config.sampling.eps, decay=self.config.sampling.decay, temperature=self.config.sampling.temperature).detach().cpu()
        else:
            samples = anneal_langevin_ebm(test_model, init_samples, 0, sigmas,
                                        n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                        final_only=self.config.sampling.final_only, denoise=self.config.sampling.denoise,
                                        verbose=self.config.sampling.verbose).detach().cpu()
        plt.figure(figsize=(16,12))
        for i in range(10): # num of samples
            plt.scatter(samples[:, i, 0], samples[:, i, 1], color=plt.get_cmap("tab20")(i), edgecolor='white', s=150)
            deltas = (samples[:, i][1:] - samples[:, i][:-1])
            deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
            for j, arrow in enumerate(deltas):
                plt.arrow(samples[j, i, 0], samples[j, i, 1], arrow[0], arrow[1], width=1e-5, head_width=2e-3, color=plt.get_cmap("tab20")(i), linewidth=3)
        plt.xlim(-xy_lim*2, xy_lim*2)
        plt.ylim(-xy_lim*2, xy_lim*2)
        plt.plot()
        plt.savefig(os.path.join(self.args.plt_fifure_path, f"sampling/pseudo_sampleing_{epoch}.png"))
        plt.close()
        # 最終サンプルだけ抽出
        final_pseudo_samples = samples[-1]
        plt.figure(figsize=(16,12))
        plt.scatter(*final_pseudo_samples.T.cpu(), alpha=0.2, color='blue', s=40)
        plt.xlim(-xy_lim*2, xy_lim*2)
        plt.ylim(-xy_lim*2, xy_lim*2)
        plt.plot()
        plt.savefig(os.path.join(self.args.plt_fifure_path, f"samples/pseudo_sample_{epoch}.png"))
        plt.close()
        np.save(os.path.join(self.args.ckpt_model_path, f'final_pseudo_samples_{epoch}.npy'), final_pseudo_samples.numpy())

'''
# grad(log p(x)) for Unconditional
xx = xx.detach().requires_grad_(True)
xx_logits = jem_model(xx, xx_labels, None)
# log_probs = F.log_softmax(xx_logits, dim=-1)
scores = torch.autograd.grad(xx_logits.logsumexp(1).sum(), xx, create_graph=True)[0]
scores = scores.detach().cpu()
# L2 norm
scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
plot_scores(all_data.T.cpu(), xx.T.detach().cpu(), scores_log1p.T.cpu(),
            os.path.join(self.args.plt_fifure_path, f"uncond_scores/scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

# grad(log p(x|y=0)) for conditional
xx = xx.detach().requires_grad_(True)
xx_logits = jem_model(xx, xx_labels, None)
condition = torch.ones(xx.shape[0], dtype=int).cuda()
scores = torch.autograd.grad(torch.gather(xx_logits, 1, condition[:, None]).sum(), xx, create_graph=True)[0].detach().cpu()
# L2 norm
scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
plot_scores(all_data.T.cpu(), xx.T.detach().cpu(), scores_log1p.T.cpu(),
            os.path.join(self.args.plt_fifure_path, f"cond_scores/scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
'''