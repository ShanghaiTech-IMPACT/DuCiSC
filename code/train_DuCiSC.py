import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

def get_current_consistency_weight(epoch, consistency=1.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, 40.0)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def get_eta_time(start_time, iter_num, total_iter):
    elapsed_time = time.time() - start_time
    estimated_time = (elapsed_time / iter_num) * (total_iter - iter_num)
    hours, rem = divmod(estimated_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='Pancreas', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='DuCiSC', help='exp_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=12, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--deepsup', type=bool, default=True, help='whether deep supervision')


parser.add_argument('--lamda_teacher', type=float, default=0.3, help='weight for consistency with teacher net')
parser.add_argument('--lamda_proto', type=float, default=0.1, help='weight for proto loss')
parser.add_argument('--num_proto', type=int, default=4, help='number of prototypes calculated for proto loss')
parser.add_argument('--lamda_mix', type=float, default=0.1, help='weight for mix loss')
parser.add_argument('--lamda_mix_proto', type=float, default=0.1, help='weight for mix proto loss')

args = parser.parse_args()

alpha = 0.99
num_classes = 2
dynamic_threshold_bg = [1/num_classes for i in range(4)]
dynamic_threshold_fg = [1/num_classes for i in range(4)]
plt_bg, plt_fg = {}, {}



snapshot_path = f'{args.root_path}/results/{args.dataset_name}_{args.exp}_{args.labelnum}_labeled_{args.base_lr}_{alpha}/{args.num_proto}_{args.lamda_proto}_{args.lamda_teacher}_{args.lamda_mix}_{args.lamda_mix_proto}'

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path+'data/Pancreas'
    args.max_samples = 62
elif args.dataset_name == "Nurves":
    patch_size = (160, 128, 112)
    args.root_path = args.root_path+'data/Nurves'
    args.max_samples = 90
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

def convert_to_soft_label(deep_label):
    foreground_prob = deep_label.unsqueeze(1) 
    background_prob = 1 - foreground_prob  #[batch_size, 1, W, H, D]
    soft_label = torch.cat([background_prob, foreground_prob], dim=1)  # [batch_size, 2, W, H, D]
    return soft_label

def custom_cross_entropy(output, target, certain_index=None):
    target = convert_to_soft_label(target)
    log_output = F.log_softmax(output, dim=1)
    if certain_index is None:
        loss = F.kl_div(log_output, target, reduction="mean")
    else:
        # certain_index: [B, W, H, D], target: [B, 2, W, H, D]
        certain_index = certain_index.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        loss = F.kl_div(log_output, target, reduction="none")[certain_index].mean()
    return loss

def get_loss(outputs, label_batch, deepsup, if_pseudo=False, if_soft=False, uncertain_area_all=None):
    if deepsup:
        loss_seg, loss_seg_dice = 0, 0
        for i in range(len(outputs)):
            output = outputs[i] # B C W H D
            if if_pseudo: # pseudo label
                deep_label = label_batch[i].long()
            elif if_soft: # soft label
                deep_label = nn.functional.interpolate(label_batch.float().unsqueeze(1), size=output.shape[2:], mode='trilinear').float()[:,0,:,:,:]
            else: # gt label
                deep_label = nn.functional.interpolate(label_batch.float().unsqueeze(1), size=output.shape[2:], mode='nearest').long()[:,0,:,:,:]
            # ce loss
            if if_pseudo: # pseudo label
                if uncertain_area_all is None:
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label)
                else:
                    certain_index = (uncertain_area_all[i]==0)
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label, reduction='none')[certain_index].mean()
            elif if_soft: # soft label
                if uncertain_area_all is None:
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label)
                else:
                    certain_index = (uncertain_area_all[i]==0)
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label, certain_index)
            else: # gt label
                loss_seg = loss_seg + F.cross_entropy(output, deep_label)
            
            # dice loss
            output_soft = F.softmax(output, dim=1)
            if uncertain_area_all is not None:
                certain_index = (uncertain_area_all[i]==0)
                loss_seg_dice = loss_seg_dice + dice_loss(output_soft[:, 1, :, :, :][certain_index], deep_label[certain_index])
            else:
                loss_seg_dice = loss_seg_dice + dice_loss(output_soft[:, 1, :, :, :], deep_label)
        loss_seg_dice = loss_seg_dice / len(outputs)
        loss_seg = loss_seg / len(outputs)
        return loss_seg, loss_seg_dice
    else:
        loss_seg = F.cross_entropy(outputs, label_batch)
        outputs_soft = F.softmax(outputs, dim=1)
        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
        return loss_seg, loss_seg_dice

def get_prototype(features, masks, argmax=False, if_bg=True, if_soft=False):
        prototypes = []
        for i, feature in enumerate(features):
            if argmax: # unlabeled data
                tmp_mask = masks[-(i+1)]
                tmp_mask = torch.argmax(tmp_mask, dim=1).unsqueeze(1)
                # print(i,len(features), feature.shape, tmp_mask.shape, masks[-(i+1)].shape)
            elif if_soft: # soft label
                tmp_mask = nn.functional.interpolate(masks.float().unsqueeze(1), size=feature.shape[2:], mode='trilinear').float()
            else: # gt label
                tmp_mask = nn.functional.interpolate(masks.float().unsqueeze(1), size=feature.shape[2:], mode='nearest').long()
            feature_tgt = feature * tmp_mask # B C W H D
            feature_tgt = feature_tgt.sum(dim=(2, 3, 4)) / tmp_mask.sum(dim=(2, 3, 4))
            proto = torch.mean(feature_tgt, dim=0)
            if if_bg:
                tmp_mask = 1 - tmp_mask
                feature_bg = feature * tmp_mask
                feature_bg = feature_bg.sum(dim=(2, 3, 4)) /tmp_mask.sum(dim=(2, 3, 4))
                feature_bg = torch.mean(feature_bg, dim=0)
                proto = torch.cat([feature_bg.unsqueeze(0),proto.unsqueeze(0) ], dim=0)
            # print(i, proto.shape)
            prototypes.append(proto)
        return prototypes

def get_prototype_loss(prototypes, prototypes_unlabeled, num_prototypes):
    # reverse prototypes and prototypes_unlabeled
    prototypes = prototypes[::-1]
    prototypes_unlabeled = prototypes_unlabeled[::-1]
    loss_prototype = []
    for i in range(num_prototypes):
        prototype = prototypes[i]
        prototype_unlabeled = prototypes_unlabeled[i]
        loss_tmp = F.mse_loss(prototype, prototype_unlabeled)
        if not torch.isnan(loss_tmp):
            loss_prototype.append(loss_tmp)
    if len(loss_prototype) > 0:
        loss_prototype = sum(loss_prototype) / len(loss_prototype)
    else:
        loss_prototype = 0
    return loss_prototype

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data
    return ema_model

if __name__ == "__main__":
    
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))
    writer = SummaryWriter(snapshot_path+'/log')
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = net_factory(net_type=args.exp, in_chns=1, class_num=num_classes, mode="train")
    model_teacher = net_factory(net_type=args.exp, in_chns=1, class_num=num_classes, mode="train")
    for param in model_teacher.parameters():
        param.detach_()

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    elif args.dataset_name == "Nurves":
        db_train = Nurves(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.1)
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    best_jc = 0
    best_hd = 100
    best_asd = 100
    start_time = time.time()
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = range(max_epoch)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_labeled, volume_unlabeled = volume_batch[:labeled_bs], volume_batch[labeled_bs:]
            model.train()
            outputs_labeled, features_labeled = model(volume_labeled, args.deepsup)
            outputs_unlabeled, features_unlabeled = model(volume_unlabeled, args.deepsup)
            loss_seg, loss_seg_dice = get_loss(outputs_labeled, label_batch[:labeled_bs], args.deepsup)
            loss_supervised = (loss_seg + loss_seg_dice) / 2

            # teacher net 
            outputs_teacher, features_teacher = model_teacher(volume_unlabeled, args.deepsup)
            pseudo_label = [torch.argmax(output, dim=1) for output in outputs_teacher]
            # dynamic_threshold 
            outputs_teacher_soft = [F.softmax(output, dim=1).detach().cpu() for output in outputs_teacher]
            pseudo_label_bg = [(output_soft[:, 0, :, :, :] > dynamic_threshold_bg[i]).long() for i,output_soft in enumerate(outputs_teacher_soft)]
            pseudo_label_fg = [(output_soft[:, 1, :, :, :] > dynamic_threshold_fg[i]).long() for i,output_soft in enumerate(outputs_teacher_soft)]
            uncertain_area_all = []
            for i in range(len(outputs_teacher_soft)):
                uncertain_area = (pseudo_label_bg[i] + pseudo_label_fg[i]) == 0
                uncertain_area_all.append(uncertain_area)
            loss_seg_pseudo, loss_seg_dice_pseudo = get_loss(outputs_unlabeled, pseudo_label, args.deepsup, if_pseudo=True, uncertain_area_all=uncertain_area_all)
            loss_consistency = (loss_seg_pseudo + loss_seg_dice_pseudo) / 2

            prototype_labeled = get_prototype(features_labeled, label_batch[:labeled_bs])
            prototype_unlabeled = get_prototype(features_unlabeled, outputs_teacher, argmax=True)
            loss_proto = get_prototype_loss(prototype_labeled, prototype_unlabeled, args.num_proto)

            # volume fusion
            fused_volume, fused_label, mixed_features = [], [], []
            fused_weight_chosen_list = []
            for i in range(labeled_bs):
                # randome select a fused wegiht between [0.25, 0.75] from uniform distribution
                fused_weight = random.uniform(0.25, 0.75)
                fused_volume.append(fused_weight*volume_labeled[i] + (1-fused_weight)*volume_unlabeled[i])
                fused_label.append(fused_weight*label_batch[i] + (1-fused_weight)*pseudo_label[-1][i])
                fused_weight_chosen_list.append(fused_weight)
            

            fused_volume = torch.stack(fused_volume, dim=0).cuda()
            fused_label = torch.stack(fused_label, dim=0).cuda()
            outputs_fused, features_fused = model(fused_volume, args.deepsup)
            loss_mix_seg, loss_mix_seg_dice = get_loss(outputs_fused, fused_label, args.deepsup, if_soft=True, uncertain_area_all=uncertain_area_all)
            loss_mix = (loss_mix_seg + loss_mix_seg_dice) / 2

            prototype_mix = get_prototype(features_fused, fused_label, if_soft=True)
            loss_mix_proto = get_prototype_loss(prototype_labeled, prototype_mix, args.num_proto)

            loss = loss_supervised / loss_supervised.detach() 
            if iter_num >= 2000: # warm-up
                if args.lamda_teacher != 0.0:
                    if not torch.isnan(loss_consistency):
                        loss = loss +  loss_consistency / loss_consistency.detach() * args.lamda_teacher # get_current_consistency_weight(iter_num//150, args.lamda_teacher)
                if args.lamda_mix != 0.0:
                    if not torch.isnan(loss_mix):
                        loss = loss + loss_mix / loss_mix.detach() * args.lamda_mix # get_current_consistency_weight(iter_num//150, args.lamda_mix)
                if loss_proto!=0 and (not torch.isnan(loss_proto)):
                    loss = loss + args.lamda_proto * loss_proto / loss_proto.detach()
                if loss_mix_proto!=0 and (not torch.isnan(loss_mix_proto)):
                    loss = loss + args.lamda_mix_proto * loss_mix_proto / loss_mix_proto.detach()
            
                # eam update the threshold
                mean_prob_bg, mean_prob_fg = [], []
                current_unlabeled_soft = [F.softmax(output, dim=1).cpu() for output in outputs_unlabeled]
                for i in range(len(current_unlabeled_soft)):
                    index_gt_fg = (pseudo_label[i] == 1).cpu()
                    index_gt_bg = (pseudo_label[i] == 0).cpu()
                    if index_gt_fg.sum() > 0:
                        tmp = current_unlabeled_soft[i][:, 1, :, :, :][index_gt_fg].mean().item()
                        mean_prob_fg.append(tmp)
                    else:
                        mean_prob_fg.append(0)
                    if index_gt_bg.sum() > 0 :
                        tmp = current_unlabeled_soft[i][:, 0, :, :, :][index_gt_bg].mean().item()
                        mean_prob_bg.append(tmp)
                    else:
                        mean_prob_bg.append(0)
                # update the threshold
                for i in range(len(dynamic_threshold_bg)):
                    if mean_prob_bg[i] != 0:
                        dynamic_threshold_bg[i] = dynamic_threshold_bg[i] * alpha + (1-alpha) * mean_prob_bg[i]
                        dynamic_threshold_bg[i] = max(0.5, dynamic_threshold_bg[i])
                        dynamic_threshold_bg[i] = min(0.95, dynamic_threshold_bg[i])
                    if mean_prob_fg[i] != 0:
                        dynamic_threshold_fg[i] = dynamic_threshold_fg[i] * alpha + (1-alpha) * mean_prob_fg[i]
                        dynamic_threshold_fg[i] = max(0.5, dynamic_threshold_fg[i])
                        dynamic_threshold_fg[i] = min(0.95, dynamic_threshold_fg[i])
                    plt_bg[iter_num] = dynamic_threshold_bg
                    plt_fg[iter_num] = dynamic_threshold_fg
            loss_all = loss_supervised + args.lamda_teacher * loss_consistency + args.lamda_mix * loss_mix + args.lamda_proto * loss_proto + args.lamda_mix_proto * loss_mix_proto
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            lr_ = optimizer.param_groups[0]['lr']
            update_ema_variables(model, model_teacher, 0.99, iter_num)
            iter_num += 1
            writer.add_scalar('train/loss', loss_all.item(), iter_num)
            writer.add_scalar('train/loss_seg', loss_seg.item(), iter_num)
            writer.add_scalar('train/loss_seg_dice', loss_seg_dice.item(), iter_num)
            writer.add_scalar('train/loss_proto', loss_proto, iter_num)
            writer.add_scalar('train/loss_seg_pseudo', loss_seg_pseudo.item(), iter_num)
            writer.add_scalar('train/loss_seg_dice_pseudo', loss_seg_dice_pseudo.item(), iter_num)
            writer.add_scalar('train/loss_mix', loss_mix.item(), iter_num)
            writer.add_scalar('train/loss_mix_proto', loss_mix_proto, iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/threshold_bg_0', dynamic_threshold_bg[0], iter_num)
            writer.add_scalar('train/threshold_bg_1', dynamic_threshold_bg[1], iter_num)
            writer.add_scalar('train/threshold_bg_2', dynamic_threshold_bg[2], iter_num)
            writer.add_scalar('train/threshold_bg_3', dynamic_threshold_bg[3], iter_num)
            writer.add_scalar('train/threshold_fg_0', dynamic_threshold_fg[0], iter_num)
            writer.add_scalar('train/threshold_fg_1', dynamic_threshold_fg[1], iter_num)
            writer.add_scalar('train/threshold_fg_2', dynamic_threshold_fg[2], iter_num)
            writer.add_scalar('train/threshold_fg_3', dynamic_threshold_fg[3], iter_num)
            
            if iter_num % 200 == 0:
                logging.info(f'iteration {iter_num}: loss: {loss_all.item():.4f}, loss_seg: {loss_seg.item():.4f}, loss_seg_dice: {loss_seg_dice.item():.4f}, loss_proto: {loss_proto:.4f}, loss_seg_pseudo: {loss_seg_pseudo.item():.4f}, loss_seg_dice_pseudo: {loss_seg_dice_pseudo.item():.4f}, loss_mix: {loss_mix.item():.4f}, loss_mix_proto: {loss_mix_proto:.4f}, lr: {lr_:.5f}, eta: {get_eta_time(start_time, iter_num, max_iterations)}')
                logging.info(f"{iter_num} dynamic_threshold_bg: {[round(i, 5) for i in dynamic_threshold_bg]}, dynamic_threshold_fg: {[round(i, 5) for i in dynamic_threshold_fg]} with uncertain area: {[i.sum() for i in uncertain_area_all]}")
                
    
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name =="LA":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Pancreas')
                elif args.dataset_name =="Nurves":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Nurves')
                
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  f'{args.exp}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"save best model to {save_mode_path}")

                logging.info(f"iter {iter_num}, dice_sample: [{dice_sample*100:.2f}/{best_dice*100:.2f}]")
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            break
    ema_data = {"bg": plt_bg, "fg": plt_fg}
    import pickle
    with open(os.path.join(snapshot_path, 'ema_data.pkl'), 'wb') as f:
        pickle.dump(ema_data, f)