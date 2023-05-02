import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchtuples as tt
import numpy as np
from loss_surv import *
from sksurv.metrics import concordance_index_censored
from file_and_folder_operations import maybe_mkdir_p
class debias_leaner(object):
    def __init__(self, img_in_features, num_nodes=[1024, 512, 128], swap_augment=False,exp_dir='', run_name='img_feat', device='cuda:0', group='race'):
        super(debias_leaner, self).__init__()

        self.exp_dir = exp_dir
        self.save_feat_dir = os.path.join(self.exp_dir, 'saved_feats_surv')
        self.save_feat_id_dir = os.path.join(self.exp_dir, 'saved_feats_id')
        self.save_feat_origin_dir = os.path.join(self.exp_dir, 'saved_feats_origin')
        maybe_mkdir_p(self.save_feat_dir)
        maybe_mkdir_p(self.save_feat_id_dir)
        maybe_mkdir_p(self.save_feat_origin_dir)
        self.result_dir = self.exp_dir + '/checkpoints/'
        maybe_mkdir_p(self.result_dir)
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(self.exp_dir+'/runs/')
        self.run_name = run_name
        self.swap_augment = swap_augment
        lr_id = 1e-3
        lr_surv = 1e-3
        weight_decay = 0.0005
        self.num_classes = 2
        self.ema_alpha = 0.7
        self.epochs = 1000
        lr_decay_step_id = [1500]
        lr_decay_step_surv = [15500]
        lr_gamma = 0.1
        self.device = torch.device(device)
        self.swap_start_eopch = 50
        self.surv_start_eopch = 0
        self.ID_start_eopch = 0
        self.lambda_swap = 0.8
        self.lambda_surv = 0.5
        self.early_stop_epoch = 600
        self.model_i = MLP_DISENTANGLE(in_features=img_in_features, num_nodes=num_nodes).to(self.device)
        self.model_b = MLP_DISENTANGLE(in_features=img_in_features, num_nodes=num_nodes).to(self.device)

        self.optimizer_b = optim.AdamW(
                self.model_b.parameters(),
                lr=lr_surv,
                weight_decay=weight_decay,
            )

        self.optimizer_i = optim.AdamW(
                self.model_i.parameters(),
                lr=lr_id,
                weight_decay=weight_decay,
            )
        self.scheduler_b = optim.lr_scheduler.MultiStepLR(self.optimizer_b, milestones=lr_decay_step_surv, gamma=lr_gamma)
        self.scheduler_i = optim.lr_scheduler.MultiStepLR(self.optimizer_i, milestones=lr_decay_step_id, gamma=lr_gamma)
        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion_surv = CoxPHLoss()
        self.criterion_id = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion_id = GeneralizedCELoss(q=0.7)
        self.best_valid_loss = 1.
        self.best_valid_loss_id_i = 1.
        self.best_valid_loss_surv = 10.
        self.best_valid_acc_surv = 0.
        self.best_valid_acc_id_i = 0.

    def train(self, tarin_data, val_data):
        print(self.run_name + ' training ',)
        train_loader = make_dataloader(tarin_data, shuffle=True)
        val_loader = make_dataloader(val_data, shuffle=True)
        id_array = tarin_data[-2]
        sample_loss_ema_b = EMA(torch.LongTensor(id_array), alpha=self.ema_alpha)
        sample_loss_ema_i = EMA(torch.LongTensor(id_array), alpha=self.ema_alpha)

        early_stop_count = 0
        train_iter = iter(train_loader)
        for step in range(self.epochs):
            
            try:
                index, feats, id_label, surv_label = next(train_iter)
            except:
                train_iter = iter(train_loader)
                index, feats, id_label, surv_label = next(train_iter)

            img_feat = feats[0].to(self.device)
            # text_feat = feats[1].to(self.device)
            id_label = id_label.to(self.device)
            surv_label = tt.tuplefy(surv_label).to_device(self.device)
            index = torch.Tensor(index).long().to(self.device)
            
            z_i = self.model_i.extract(img_feat) # intrinsic feature, relate to race or ethnicity
            z_b = self.model_b.extract(img_feat) # bias feature

            # Gradients of z_b are not backpropagated to z_i (and vice versa) in order to guarantee disentanglement of representation.
            z_align = torch.cat((z_b.detach(), z_i), dim=1)
            z_conflict = torch.cat((z_b, z_i.detach()), dim=1) 
            
            # Prediction using z=[z_i, z_b]
            pred_align = self.model_i.id_fc(z_align)
            pred_conflict = self.model_b.id_fc(z_conflict)
            loss_dis_align = self.criterion_id(pred_align, id_label).detach()
            loss_dis_conflict = self.criterion_id(pred_conflict, id_label).detach()
            
            # ***modify*** "bias" not relate to race or ethnicity
            pred_surv = self.model_b.surv_fc(z_b)
            pred_surv = tt.tuplefy(pred_surv)

            lambda_surv = 0
            if step > self.surv_start_eopch:
                lambda_surv = self.lambda_surv
            loss_dis_surv = self.criterion_surv(*pred_surv, *surv_label)
            
            # class-wise normalize
            sample_loss_ema_i.update(loss_dis_align, index)
            sample_loss_ema_b.update(loss_dis_conflict, index)
            loss_dis_align = sample_loss_ema_i.parameter[index].clone().detach()
            loss_dis_conflict = sample_loss_ema_b.parameter[index].clone().detach()
            loss_dis_align = loss_dis_align.to(self.device)
            loss_dis_conflict = loss_dis_conflict.to(self.device)
            for c in range(self.num_classes):
                class_index = torch.where(id_label == c)[0].to(self.device)
                max_loss_align = sample_loss_ema_i.max_loss(c)
                max_loss_conflict = sample_loss_ema_b.max_loss(c)
                loss_dis_align[class_index] /= max_loss_align
                loss_dis_conflict[class_index] /= max_loss_conflict
                
            loss_weight = loss_dis_conflict / (loss_dis_conflict + loss_dis_align + 1e-8)
            loss_dis_align = self.criterion_id(pred_align, id_label) * loss_weight.to(self.device)
            loss_dis_conflict = self.bias_criterion_id(pred_conflict, id_label)

                
            loss_dis = loss_dis_align.mean() + loss_dis_conflict.mean()
            loss = loss_dis + loss_dis_surv * lambda_surv

            # before feature-level augmentation
            loss_swap_align = torch.tensor([0]).float()
            loss_swap_conflict = torch.tensor([0]).float()
            lambda_swap = 0
            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if self.swap_augment:
                if step > self.swap_start_eopch:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = id_label[indices]     # y tilde

                    # # Prediction using z_swap=[z_l, z_b tilde]
                    # # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                    z_mix_align = torch.cat((z_b_swap.detach(), z_i), dim=1)
                    z_mix_conflict = torch.cat((z_b_swap, z_i.detach()), dim=1)

                    # # Prediction using z_swap
                    pred_mix_align = self.model_i.id_fc(z_mix_align)
                    pred_mix_conflict = self.model_b.id_fc(z_mix_conflict)

                    loss_swap_align = self.criterion_id(pred_mix_align, id_label) * loss_weight.to(self.device)
                    loss_swap_conflict = self.bias_criterion_id(pred_mix_conflict, label_swap)
                    lambda_swap = self.lambda_swap

            loss_swap = loss_swap_align.mean() + loss_swap_conflict.mean()
            loss = loss + lambda_swap * loss_swap

            self.optimizer_i.zero_grad()
            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_i.step()
            self.optimizer_b.step()

            self.scheduler_b.step()
            self.scheduler_i.step()
            
            self.board_ours_loss(
                step=step,
                loss_dis_align=loss_dis_align.mean(),
                loss_dis_conflict=loss_dis_conflict.mean(),
                loss_swap_align=loss_swap_align.mean(),
                loss_swap_conflict=loss_swap_conflict.mean(),
                lambda_swap=lambda_swap,
                loss_dis_surv=loss_dis_surv,
            )
            early_stop_count = self.board_ours_acc(step, val_loader, early_stop_count)
            
            if early_stop_count == self.early_stop_epoch:
                print(" Early stopping")
                break
        
    def evaluate_ours(self, data_loader, model='label'):
        val_iter = iter(data_loader) # all data in a batch
        index, feats, id_label, surv_label = next(val_iter)
        surv_label_np = surv_label.to_numpy()
        img_feat = feats[0].to(self.device)
        id_label = id_label.to(self.device)

        # pred id label
        pred_id_i, pred_label_i = self.predict_id(img_feat, model='label',)
        correct = (pred_id_i == id_label).float()
        accs_id_i = correct.mean().cpu().numpy()
        loss_id_i = self.criterion_id(pred_label_i, id_label).mean()
        # pred surv
        z_b = self.model_b.extract(img_feat) # intrinsic feature
        pred_surv = self.model_b.surv_fc(z_b)
        risk_ = pred_surv.detach().cpu().numpy() # for testing
        # pred cphloss
        pred_surv = tt.tuplefy(pred_surv)
        surv_label = tt.tuplefy(surv_label).to_device(self.device)
        loss_surv = self.criterion_surv(*pred_surv, *surv_label)

        # pred c_index
        c_index = concordance_index_censored(surv_label_np[1].astype(bool), surv_label_np[0], risk_[:,0])[0]
        accs_surv = c_index
        
        self.model_b.train()
        self.model_i.train()

        return accs_surv, loss_surv, loss_id_i, accs_id_i# , accs_id_b
    
    def predict_id(self, feat, model='label',):
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)
        feat = feat.to(self.device)
        z_i = self.model_i.extract(feat) # intrinsic feature
        z_b = self.model_b.extract(feat) # bias feature      
        z_origin = torch.cat((z_b, z_i), dim=1)
        if model == 'bias':
            pred_label = self.model_b.id_fc(z_origin)
        else:
            pred_label = self.model_i.id_fc(z_origin)
        pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
        return pred, pred_label
    
    # predict surv
    def predict(self, feat, load_best=True, save_feat_name=None):
        if load_best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
            state_dict = torch.load(model_path)
            self.model_b.load_state_dict(state_dict['state_dict'])
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)
        feat = feat.to(self.device)

        z_b = self.model_b.extract(feat) # bias feature      
        pred_surv = self.model_b.surv_fc(z_b)
        risk_ = pred_surv.detach().cpu().numpy() # for testing
        if save_feat_name!=None:
            self.save_feat_surv(z_b, save_feat_name)
            # get id features
            model_path = os.path.join(self.result_dir, "best_model_i.th")
            state_dict = torch.load(model_path)
            self.model_i.load_state_dict(state_dict['state_dict'])
            z_i = self.model_i.extract(feat)
            self.save_feat_id(z_i, save_feat_name)
            z_origin = torch.cat((z_b, z_i), dim=1)
            self.save_feat_origin(z_origin, save_feat_name)
        return risk_
    
    def save_feat_surv(self, feat, save_feat_name):
        if isinstance(feat, torch.Tensor):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = feat
        np.save(os.path.join(self.save_feat_dir, save_feat_name+'.npy'), feat_np)
        return    
    
    def save_feat_id(self, feat, save_feat_name):
        if isinstance(feat, torch.Tensor):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = feat
        np.save(os.path.join(self.save_feat_id_dir, save_feat_name+'.npy'), feat_np)
        return

    def save_feat_origin(self, feat, save_feat_name):
        if isinstance(feat, torch.Tensor):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = feat
        np.save(os.path.join(self.save_feat_origin_dir, save_feat_name+'.npy'), feat_np)
        return

    def board_ours_acc(self, step, data_loader, early_stop_count):
        valid_accs_surv, valid_loss_surv, valid_loss_id_i, valid_accs_id_i = self.evaluate_ours(data_loader, model='label')
        valid_loss = valid_loss_id_i + valid_loss_surv * self.lambda_surv
        if step > self.surv_start_eopch:
            if valid_accs_surv >= self.best_valid_acc_surv:
                self.best_valid_acc_surv = valid_accs_surv
            if valid_loss_surv < self.best_valid_loss_surv:
                self.best_valid_loss_surv = valid_loss_surv
            #     self.save_ours(step, best=True)
            #     early_stop_count = 0
            # else:
            #     early_stop_count += 1
        if step > self.ID_start_eopch:
            if valid_accs_id_i >= self.best_valid_acc_id_i:
                self.best_valid_acc_id_i = valid_accs_id_i
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_ours(step, best=True)
                early_stop_count = 0
            else:
                early_stop_count += 1
            if valid_loss_id_i < self.best_valid_loss_id_i:
                self.best_valid_loss_id_i = valid_loss_id_i
        self.writer.add_scalar(f"acc/valid_loss_id_i", valid_loss_id_i, step)
        self.writer.add_scalar(f"acc/best_valid_loss_id_i", self.best_valid_loss_id_i, step)
        self.writer.add_scalar(f"acc/valid_accs_id_i", valid_accs_id_i, step)
        self.writer.add_scalar(f"acc/valid_accs_surv", valid_accs_surv, step)
        self.writer.add_scalar(f"acc/best_valid_accs_surv", self.best_valid_acc_surv, step)
        self.writer.add_scalar(f"acc/valid_loss_surv", valid_loss_surv, step)
        self.writer.add_scalar(f"acc/best_valid_loss_surv", self.best_valid_loss_surv, step)
        self.writer.add_scalar(f"acc/valid_loss", valid_loss, step)
        return early_stop_count
    
        
    def board_ours_loss(self, step, loss_dis_align=-1, loss_dis_conflict=-1, 
                        loss_swap_align=-1, loss_swap_conflict=-1,
                        lambda_swap=-1, loss_dis_surv=-1, ):
        
        self.writer.add_scalar(f"loss/loss_dis_align",  loss_dis_align, step)
        self.writer.add_scalar(f"loss/loss_dis_conflict",     loss_dis_conflict, step)
        self.writer.add_scalar(f"loss/loss_dis_surv",     loss_dis_surv, step)
        self.writer.add_scalar(f"loss/loss_swap_align", loss_swap_align, step)
        self.writer.add_scalar(f"loss/loss_swap_conflict",    loss_swap_conflict, step)
        self.writer.add_scalar(f"loss/loss",               (loss_dis_align + loss_dis_conflict) + lambda_swap * (loss_swap_align + loss_swap_conflict), step)

    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_i.th")
        else:
            model_path = os.path.join(self.result_dir, "model_i_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_i.state_dict(),
            'optimizer': self.optimizer_i.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        
        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        # print(f'{step} model saved ...')

class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class EMA:
    def __init__(self, label, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()


class MLP_DISENTANGLE(nn.Module):
    def __init__(self, in_features = 1024, num_nodes = [1024, 512, 128]):
        super(MLP_DISENTANGLE, self).__init__()
        self.feature = MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=None,
                                  batch_norm=True, dropout=0.1, output_bias=False, 
                                  output_activation=None)
        
        self.id_fc = nn.Sequential(
            # nn.Linear(num_nodes[-1]*2, 2),
            nn.Linear(num_nodes[-1]*2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2, bias=False),
            nn.ReLU(inplace=True),
            torch.nn.Softmax(dim=1))
        self.surv_fc = nn.Sequential(
            nn.Linear(num_nodes[-1], 1),
            torch.nn.Sigmoid())
        
    def extract(self, x):
        x = x.view(x.size(0), -1)
        feat = self.feature(x)
        return feat

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) 
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

    
class MLPVanilla(nn.Module):
    def __init__(self, in_features, num_nodes, out_features=None, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True, 
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = tt.tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        if out_features:
            net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
    
class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


# ------------------------------- data loader ----------------------------
from torch.utils.data.dataset import Dataset

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def make_dataloader(tarin_data, shuffle, num_workers=0, to_tensor=True, ):
    img_feat_dict_tr, id_dict_tr, y_dict_tr = tarin_data
    batch_size = img_feat_dict_tr.shape[0]
    data = ((img_feat_dict_tr,), id_dict_tr, y_dict_tr)  # TODO
    if to_tensor:
        data = tt.tuplefy(data).to_tensor().to_device('cuda:0')
    else:
        data = tt.tuplefy(data)
    dataset = tt.data.DatasetTuple(*data)
    dataset = IdxDataset(dataset)
    DataLoader = tt.data.DataLoaderBatch

    # DataLoader = torch.utils.data.DataLoader
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, )
    return dataloader




