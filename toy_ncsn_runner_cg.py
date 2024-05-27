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


__all__ = ['CG_NCSNRunner']

# ----------------------------------------
# Model
# ----------------------------------------
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes, class_labels):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        # Noise scale embed
        self.embed_scale = nn.Embedding(num_classes, num_out)
        self.embed_scale.weight.data.uniform_()
        # Class embed
        self.embed_class = nn.Embedding(class_labels, num_out)
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
    def __init__(self, input_dim, num_classes, data_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes, data_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes, data_classes)
        self.lin3 = nn.Linear(128, input_dim)
    
    def forward(self, x, y, c):
        x = F.softplus(self.lin1(x, y, c))
        x = F.softplus(self.lin2(x, y, c))
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
        self.lin1 = ConditionalLinear(input_dim, 128, num_classes, data_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes, data_classes)
        # self.lin3 = nn.Linear(128, input_dim)
        self.lin4 = nn.Linear(128, data_classes)
    
    def forward(self, x, y, c):
        x = F.softplus(self.lin1(x, y, None))
        x = F.softplus(self.lin2(x, y, None))
        # return self.lin3(x), self.lin4(x)
        return self.lin4(x)

# Choose model
def get_model(config):
    if config.training.mode == 'joint_learning':
        model = JointLearningModel(config.input_dim,config.model.num_classes, config.data.d_classes)
        return model.to(config.device)
    elif config.training.mode == 'clf_guided':
        gen_model = ConditionalModel(config.input_dim,config.model.num_classes, config.data.d_classes)
        clf_model = ClassifierModel(config.input_dim,config.model.num_classes, config.data.d_classes)
        return gen_model.to(config.device), clf_model.to(config.device)
    elif config.training.mode == 'clf':
        model = ClassifierModel(config.input_dim,config.model.num_classes, 2)
        return model.to(config.device)
    else:
        model = ConditionalModel(config.input_dim,config.model.num_classes)
        return model.to(config.device)

# ----------------------------------------
# Choose Dataset
# ----------------------------------------
def get_dataset(config):
    train_data = np.load(f"{config.data.dir}/train_data.npy")
    train_labels = np.load(f"{config.data.dir}/train_labels.npy")
    test_data = np.load(f"{config.data.dir}/test_data.npy")
    test_labels = np.load(f"{config.data.dir}/test_labels.npy")
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
    train_dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    dload_train_labeled = DataLoader(labeled_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_train = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    dload_test = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    return dload_train, dload_train_labeled, dload_test

# ----------------------------------------
# LOSS FUNCTION
# ----------------------------------------
def make_perturbed_samples(samples, sigmas):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    return perturbed_samples, used_sigmas, noise, labels

def anneal_dsm(scorenet, samples, sigmas, classes=None, anneal_power=2.0):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    if classes == None: # For unconditional
        scores = scorenet(perturbed_samples, labels, None)
    else: # For conditional
        scores = scorenet(perturbed_samples, labels, classes)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0), perturbed_samples, scores

def anneal_ce_loss(scorenet, samples, sigmas, classes):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    # conditionつけてlogitだしてる
    logits = scorenet(perturbed_samples, labels, None)
    acc = (logits.max(1)[1] == classes).float().mean()
    ce_loss = nn.CrossEntropyLoss()(logits, classes)
    return ce_loss, perturbed_samples, acc
'''
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
'''
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

class CG_NCSNRunner():
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
        folders = ['uncond_scores', 'cond_scores', 'grad_probs', 'CG_theory']
        for folder in folders:
            path = os.path.join(args.plt_fifure_path, folder)
            os.makedirs(path, exist_ok=True)


    def train_clf_guided_sl(self):
        train_loader, test_loader = get_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        tb_logger = self.config.tb_logger

        gen_model, clf_model = get_model(self.config)
        # score = torch.nn.DataParallel(score)
        optimizer_gen = optim.Adam(gen_model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)
        optimizer_clf = optim.Adam(clf_model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels 0 > 1 > 2 >..
        sigmas = get_sigmas(self.config)
        best_test_acc = 0.
        best_test_dsm = 2.
        for epoch in range(start_epoch, self.config.training.n_epochs):
            dsm_losses, dsm_conds, dsm_unconds, ce_losses, total_losses, train_accs, consistencies = [], [], [], [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                gen_model.train()
                clf_model.train()
                step += 1
                gen_L = 0
                clf_L = 0
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                
                # class conditional
                dsm_cond, p_samples_cond, scores_cond = anneal_dsm(gen_model, X, sigmas, classes=y, anneal_power=self.config.training.anneal_power)
                # class unconditional
                dsm_uncond, p_samples_uncond, scores_uncond = anneal_dsm(gen_model, X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)
                # classification
                ce_loss, p_samples_clf, acc = anneal_ce_loss(clf_model, X, sigmas, y)
                # Total LOSS
                gen_L += dsm_cond*0.5 + dsm_uncond*0.5
                clf_L += ce_loss
                # LOSS SPIKE CHECK
                # if ce_loss.item() > 1.0:
                #     trai_post = nn.Softmax(dim=1)(logits).detach().cpu()
                #     test_plot(l_perterbed.T.cpu(), os.path.join(self.args.test_path, f"train_div_{epoch}.png"), plot_lim=xy_lim, posterior=trai_post)

                # consistency loss
                if True:
                    if epoch % 2 == 0:
                        wathing_class = 0
                    else:
                        wathing_class = 1
                    with torch.enable_grad():
                        x_samples = X[y==wathing_class][:500].detach().requires_grad_(True)
                        x_labels = torch.randint(0, len(sigmas), (x_samples.shape[0],), device=x_samples.device)
                        x_logits = clf_model(x_samples, x_labels, None)
                        x_log_probs = F.log_softmax(x_logits, dim=-1)
                        selected = x_log_probs[range(len(x_logits)), wathing_class]
                        grad_probs = torch.autograd.grad(selected.sum(), x_samples, create_graph=True)[0]
                    # g log p(x|y) = g log p(y|x) + g log p(x)
                    consistency = ((scores_cond[y==wathing_class][:500].detach() - (scores_uncond[y==wathing_class][:500].detach() + grad_probs)) ** 2).sum(dim=-1)
                    # clf_L += consistency.mean(dim=0) * 1.0

                dsm_conds.append(dsm_cond.item())
                dsm_unconds.append(dsm_uncond.item())
                dsm_losses.append((dsm_cond.item()+dsm_uncond.item())*0.5)
                ce_losses.append(ce_loss.item())
                # total_losses.append(L.item())
                train_accs.append(acc.item())
                consistencies.append(consistency.mean().item())

                optimizer_gen.zero_grad()
                optimizer_clf.zero_grad()
                gen_L.backward()
                clf_L.backward()
                optimizer_gen.step()
                optimizer_clf.step()
            
            tb_logger.add_scalar('train/dsm_loss', np.mean(dsm_losses), global_step=epoch)
            tb_logger.add_scalar('train/cond_dsm', np.mean(dsm_conds), global_step=epoch)
            tb_logger.add_scalar('train/uncond_dsm', np.mean(dsm_unconds), global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', np.mean(ce_losses), global_step=epoch)
            tb_logger.add_scalar('train/consistency', np.mean(consistencies), global_step=epoch)
            # tb_logger.add_scalar('train/total_loss', np.mean(total_losses), global_step=epoch)
            tb_logger.add_scalar('train/acc', np.mean(train_accs), global_step=epoch)
            logging.info("epoch: {}, step: {}, dsm: {}, ce: {}, acc: {}, consistency: {}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs), np.mean(consistencies)))

            # Eval
            if epoch % 100 == 0:
                test_score = gen_model
                test_clf = clf_model
                test_score.eval()
                test_clf.eval()
                test_dsms, test_ces, test_acces, test_eces = [], [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    # 一番小さいnoiseがかかっている体で
                    test_labels = torch.randint(3, len(sigmas), (test_X.shape[0],)).cuda()
                    with torch.no_grad():
                        # class conditional
                        test_dsm_cond, _, _ = anneal_dsm(test_score, test_X, sigmas, classes=test_y, anneal_power=self.config.training.anneal_power)
                        # class unconditional
                        test_dsm_uncond, _, _ = anneal_dsm(test_score, test_X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)
                        test_dsms.append((test_dsm_cond.item()+test_dsm_uncond.item())*0.5)
                        # test data without noise (no mean w/noise?)
                        # encoderが学習しているのはnoisy data conditionつけずにlogitだしていいのか？
                        t_logits = test_clf(test_X, test_labels, None)
                        test_post = nn.Softmax(dim=1)(t_logits).detach().cpu()
                        test_acc = (t_logits.max(1)[1] == test_y).float().mean()
                        test_ce_loss = nn.CrossEntropyLoss()(t_logits, test_y)
                        test_ces.append(test_ce_loss.item())
                        test_acces.append(test_acc.item())
                # have some trouble
                test_plot(test_X.T.cpu(), os.path.join(self.args.test_path, f"test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                # test_plot(t_perturbed_data.T.cpu(), os.path.join(self.args.test_path, f"per_test_{epoch}.png"), plot_lim=xy_lim)
                tb_logger.add_scalar('test/dsm_loss', np.mean(test_dsms), global_step=epoch)
                tb_logger.add_scalar('test/ce_loss', np.mean(test_ces), global_step=epoch)
                tb_logger.add_scalar('test/acc', np.mean(test_acces), global_step=epoch)
                # tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                logging.info("epoch: {}, step: {}, test_dsm: {}, test_ce: {}, test_acc: {}"
                             .format(epoch, step, np.mean(test_dsms), np.mean(test_ces), np.mean(test_acces)))
                del test_score, test_clf, test_dsms, test_ces, test_acces, test_post
                
                # Plot
                if epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    xx_labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()

                    # P(y|x) for all samples
                    all_data = X.clone().cuda()
                    all_labels = torch.randint(0, len(sigmas), (all_data.shape[0],)).cuda()
                    all_logits = clf_model(all_data, all_labels, None)
                    all_post = nn.Softmax(dim=1)(all_logits).detach().cpu()

                    # grad(log p(x)) for Unconditional
                    scores = gen_model(xx, xx_labels, None)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"uncond_scores/scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    
                    # Posterior score
                    with torch.enable_grad():
                        xx_in_0 = xx.detach().requires_grad_(True)
                        s_logits = clf_model(xx_in_0, xx_labels, None)
                        log_probs = F.log_softmax(s_logits, dim=-1)
                        # class 0's log_probs
                        selected_0 = log_probs[range(len(s_logits)), 0]
                        grad_probs_0 = torch.autograd.grad(selected_0.sum(), xx_in_0)[0]
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), grad_probs_0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"grad_probs/grad_p(y=0lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    with torch.enable_grad():
                        xx_in_1 = xx.detach().requires_grad_(True)
                        s_logits = clf_model(xx_in_1, xx_labels, None)
                        log_probs = F.log_softmax(s_logits, dim=-1)
                        # class 1's log_probs
                        selected_1 = log_probs[range(len(s_logits)), 1]
                        grad_probs_1 = torch.autograd.grad(selected_1.sum(), xx_in_1)[0]
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), grad_probs_1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"grad_probs/grad_p(y=1lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    
                    # grad(log p(x|y=0)) for conditional
                    scores = gen_model(xx, xx_labels, torch.zeros(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c0 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/scoreslog_p(xly=0)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # grad(log p(x|y=1)) for conditional
                    scores = gen_model(xx, xx_labels, torch.ones(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c1 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/scoreslog_p(xly=1)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # # ベクトルの足し算したものを表示する
                    hikizan_c0 = grad_probs_0.cpu()+scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(y=0lx)+p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c0 = scores_log1p_c0-scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(xly=0)-p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c0 = scores_log1p_c0-grad_probs_0.cpu()
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(xly=0)-p(y=0lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

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

    def train_clf_guided_ssl(self):
        train_loader, train_labeled_loader, test_loader = get_ssl_dataset(self.config)
        self.config.input_dim = self.config.data.dim * self.config.data.channels
        xy_lim = float(self.config.data.mu) + 3.0
        tb_logger = self.config.tb_logger

        gen_model, clf_model = get_model(self.config)
        # score = torch.nn.DataParallel(score)
        optimizer_gen = optim.Adam(gen_model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)
        optimizer_clf = optim.Adam(clf_model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                        eps=self.config.optim.eps)

        start_epoch = 0
        step = 0
        # Noise labels 0 > 1 > 2 >..
        sigmas = get_sigmas(self.config)
        best_test_acc = 0.
        best_test_dsm = 2.
        for epoch in range(start_epoch, self.config.training.n_epochs):
            dsm_losses, dsm_conds, dsm_unconds, ce_losses, total_losses, train_accs, consistencies = [], [], [], [], [], [], []
            for i, (X, y) in enumerate(train_loader):
                gen_model.train()
                clf_model.train()
                step += 1
                gen_L = 0
                clf_L = 0
                X = X.to(self.config.device)
                # y = y.to(self.config.device)

                # labeled data
                labeled_X, labeled_y = train_labeled_loader.__next__()
                labeled_X, labeled_y = labeled_X.to(self.config.device), labeled_y.to(self.config.device)
                
                # class conditional
                dsm_cond, p_samples_cond, scores_cond = anneal_dsm(gen_model, labeled_X, sigmas, classes=labeled_y, anneal_power=self.config.training.anneal_power)
                # class unconditional
                dsm_uncond, p_samples_uncond, scores_uncond = anneal_dsm(gen_model, X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)
                # classification
                ce_loss, p_samples_clf, acc = anneal_ce_loss(clf_model, labeled_X, sigmas, labeled_y)
                # Total LOSS
                gen_L += dsm_cond*0.5 + dsm_uncond*0.5
                clf_L += ce_loss * self.config.training.ce_weight
                # LOSS SPIKE CHECK
                # if ce_loss.item() > 1.0:
                #     trai_post = nn.Softmax(dim=1)(logits).detach().cpu()
                #     test_plot(l_perterbed.T.cpu(), os.path.join(self.args.test_path, f"train_div_{epoch}.png"), plot_lim=xy_lim, posterior=trai_post)

                # consistency loss
                if True:
                    if epoch % 2 == 0:
                        wathing_class = 0
                    else:
                        wathing_class = 1
                    with torch.enable_grad():
                        x_samples = X.detach().requires_grad_(True)
                        x_labels = torch.randint(0, len(sigmas), (x_samples.shape[0],), device=x_samples.device)
                        x_logits = clf_model(x_samples, x_labels, None)
                        x_log_probs = F.log_softmax(x_logits, dim=-1)
                        selected = x_log_probs[range(len(x_logits)), wathing_class]
                        grad_probs = torch.autograd.grad(selected.sum(), x_samples, create_graph=True)[0]
                    with torch.no_grad():
                        buf, buf2, pseudo_scores_cond = anneal_dsm(gen_model, x_samples, sigmas, classes=torch.tensor([wathing_class]*500).cuda(), anneal_power=self.config.training.anneal_power)
                    # g log p(x|y) = g log p(y|x) + g log p(x)
                    consistency = ((pseudo_scores_cond.detach() - (scores_uncond.detach() + grad_probs)) ** 2).sum(dim=-1)
                    clf_L += consistency.mean(dim=0) * self.config.training.cr_weight

                dsm_conds.append(dsm_cond.item())
                dsm_unconds.append(dsm_uncond.item())
                dsm_losses.append((dsm_cond.item()+dsm_uncond.item())*0.5)
                ce_losses.append(ce_loss.item())
                # total_losses.append(L.item())
                train_accs.append(acc.item())
                consistencies.append(consistency.mean().item())

                optimizer_gen.zero_grad()
                optimizer_clf.zero_grad()
                gen_L.backward()
                clf_L.backward()
                optimizer_gen.step()
                optimizer_clf.step()
            
            tb_logger.add_scalar('train/dsm_loss', np.mean(dsm_losses), global_step=epoch)
            tb_logger.add_scalar('train/cond_dsm', np.mean(dsm_conds), global_step=epoch)
            tb_logger.add_scalar('train/uncond_dsm', np.mean(dsm_unconds), global_step=epoch)
            tb_logger.add_scalar('train/ce_loss', np.mean(ce_losses), global_step=epoch)
            tb_logger.add_scalar('train/consistency', np.mean(consistencies), global_step=epoch)
            # tb_logger.add_scalar('train/total_loss', np.mean(total_losses), global_step=epoch)
            tb_logger.add_scalar('train/acc', np.mean(train_accs), global_step=epoch)
            logging.info("epoch: {},\tstep: {},\tdsm: {:.5f}, ce: {:.5f}, acc: {:.5f}, consistency: {:.5f}"
                         .format(epoch, step, np.mean(dsm_losses), np.mean(ce_losses), np.mean(train_accs), np.mean(consistencies)))

            # Eval
            if epoch % 100 == 0:
                test_score = gen_model
                test_clf = clf_model
                test_score.eval()
                test_clf.eval()
                test_dsms, test_dsm_conds, test_dsm_unconds, test_ces, test_acces, test_eces = [], [], [], [], [], []
                for i, (test_X, test_y) in enumerate(test_loader):
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    test_labels = torch.randint(len(sigmas)-1, len(sigmas), (test_X.shape[0],)).cuda()
                    with torch.no_grad():
                        # class conditional
                        test_dsm_cond, _, _ = anneal_dsm(test_score, test_X, sigmas, classes=test_y, anneal_power=self.config.training.anneal_power)
                        # class unconditional
                        test_dsm_uncond, _, _ = anneal_dsm(test_score, test_X, sigmas, classes=None, anneal_power=self.config.training.anneal_power)
                        test_dsms.append((test_dsm_cond.item()+test_dsm_uncond.item())*0.5)
                        test_dsm_conds.append(test_dsm_cond.item())
                        test_dsm_unconds.append(test_dsm_uncond.item())
                        # test data without noise (no mean w/noise?)
                        # encoderが学習しているのはnoisy data conditionつけずにlogitだしていいのか？
                        t_logits = test_clf(test_X, test_labels, None)
                        test_post = nn.Softmax(dim=1)(t_logits).detach().cpu()
                        test_acc = (t_logits.max(1)[1] == test_y).float().mean()
                        test_ce_loss = nn.CrossEntropyLoss()(t_logits, test_y)
                        test_ces.append(test_ce_loss.item())
                        test_acces.append(test_acc.item())
                # have some trouble
                test_plot(test_X.T.cpu(), os.path.join(self.args.test_path, f"test_{epoch}.png"), plot_lim=xy_lim, posterior=test_post)
                # test_plot(t_perturbed_data.T.cpu(), os.path.join(self.args.test_path, f"per_test_{epoch}.png"), plot_lim=xy_lim)
                tb_logger.add_scalar('test/dsm_loss', np.mean(test_dsms), global_step=epoch)
                tb_logger.add_scalar('test/cond_dsm', np.mean(test_dsm_conds), global_step=epoch)
                tb_logger.add_scalar('test/uncond_dsm', np.mean(test_dsm_unconds), global_step=epoch)
                tb_logger.add_scalar('test/ce_loss', np.mean(test_ces), global_step=epoch)
                tb_logger.add_scalar('test/acc', np.mean(test_acces), global_step=epoch)
                # tb_logger.add_scalar('test/ece', calibration_score, global_step=epoch)
                logging.info("epoch: {},\tstep: {},\ttest_dsm: {:.5f}, test_ce: {:.5f}, test_acc: {:.5f}"
                             .format(epoch, step, np.mean(test_dsms), np.mean(test_ces), np.mean(test_acces)))
                del test_score, test_clf, test_dsms, test_ces, test_acces, test_post
                
                # Plot
                if epoch % 100 == 0:
                    xx = np.stack(np.meshgrid(np.linspace(-xy_lim, xy_lim, 50), np.linspace(-xy_lim, xy_lim, 50)), axis=-1).reshape(-1, 2)
                    xx = torch.from_numpy(xx.astype(np.float32)).clone().cuda()
                    xx_labels = torch.randint(0, len(sigmas), (xx.shape[0],)).cuda()
                    # labels = torch.ones(xx.shape[0]).long().cuda()

                    # P(y|x) for all samples
                    all_data = X.clone().cuda()
                    all_labels = torch.randint(0, len(sigmas), (all_data.shape[0],)).cuda()
                    all_logits = clf_model(all_data, all_labels, None)
                    all_post = nn.Softmax(dim=1)(all_logits).detach().cpu()

                    # grad(log p(x)) for Unconditional
                    scores = gen_model(xx, xx_labels, None)
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"uncond_scores/scoreslog_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    
                    # Posterior score
                    with torch.enable_grad():
                        xx_in_0 = xx.detach().requires_grad_(True)
                        s_logits = clf_model(xx_in_0, xx_labels, None)
                        log_probs = F.log_softmax(s_logits, dim=-1)
                        # class 0's log_probs
                        selected_0 = log_probs[range(len(s_logits)), 0]
                        grad_probs_0 = torch.autograd.grad(selected_0.sum(), xx_in_0)[0]
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), grad_probs_0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"grad_probs/grad_p(y=0lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    with torch.enable_grad():
                        xx_in_1 = xx.detach().requires_grad_(True)
                        s_logits = clf_model(xx_in_1, xx_labels, None)
                        log_probs = F.log_softmax(s_logits, dim=-1)
                        # class 1's log_probs
                        selected_1 = log_probs[range(len(s_logits)), 1]
                        grad_probs_1 = torch.autograd.grad(selected_1.sum(), xx_in_1)[0]
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), grad_probs_1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"grad_probs/grad_p(y=1lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    
                    # grad(log p(x|y=0)) for conditional
                    scores = gen_model(xx, xx_labels, torch.zeros(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c0 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c0.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/scoreslog_p(xly=0)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # grad(log p(x|y=1)) for conditional
                    scores = gen_model(xx, xx_labels, torch.ones(len(xx), dtype=int, device=self.config.device))
                    scores = scores.detach().cpu()
                    # L2 norm
                    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
                    scores_log1p_c1 = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), scores_log1p_c1.T.cpu(),
                                os.path.join(self.args.plt_fifure_path, f"cond_scores/scoreslog_p(xly=1)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)

                    # # ベクトルの足し算したものを表示する
                    hikizan_c0 = grad_probs_0.cpu()+scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(y=0lx)+p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c0 = scores_log1p_c0-scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(xly=0)-p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c0 = scores_log1p_c0-grad_probs_0.cpu()
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c0.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/0_p(xly=0)-p(y=0lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c1 = grad_probs_1.cpu()+scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c1.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/1_p(y=1lx)+p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c1 = scores_log1p_c1-scores_log1p
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c1.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/1_p(xly=1)-p(x)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
                    hikizan_c1 = scores_log1p_c1-grad_probs_1.cpu()
                    plot_scores(all_data.T.cpu(), xx.T.cpu(), hikizan_c1.T,
                                os.path.join(self.args.plt_fifure_path, f"CG_theory/1_p(xly=1)-p(y=1lx)_{epoch}.png"), plot_lim=xy_lim, posterior=all_post)
