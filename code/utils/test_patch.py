import h5py
import math
import nibabel as nib
import numpy as np
import os
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('./data/LA/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/LA/2018LA_Seg_TrainingSet/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas":
        with open('./data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    elif dataset_name == "Nurves":
        with open('./data/Nurves/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Nurves/Nurves_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    total_jc = 0.0
    total_hd = 0.0
    total_asd = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label<0] = 0
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice, jc, hd, asd = 0, 0, 0, 0
        else:
            dice = metric.binary.dc(prediction, label) # metric.binary.dc(prediction, label)
            jc = metric.binary.jc(prediction, label)
            hd = metric.binary.hd95(prediction, label)
            asd = metric.binary.asd(prediction, label)
        total_dice += dice
        total_jc += jc
        total_hd += hd
        total_asd += asd
    avg_dice = total_dice / len(image_list)
    avg_jc = total_jc / len(image_list)
    avg_hd = total_hd / len(image_list)
    avg_asd = total_asd / len(image_list)
    print(f'average dice: {avg_dice*100:.2f}, average jc: {avg_jc*100:.2f}, average hd: {avg_hd:.2f}, average asd: {avg_asd:.2f}')
    return avg_dice, avg_jc, avg_hd, avg_asd

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = {'dice' : [], 'jc': [], 'hd': [], 'asd': []}
    infer_info = []
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        name = os.path.basename(image_path.split(".")[0])
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label<0] = 0
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric['dice'].append(single_metric[0])
        total_metric['jc'].append(single_metric[1])
        total_metric['hd'].append(single_metric[2])
        total_metric['asd'].append(single_metric[3])
        if metric_detail:
            info = f'case {ith}, dice: {single_metric[0]*100:.2f}, jc: {single_metric[1]*100:.2f}, hd: {single_metric[2]:.2f}, asd: {single_metric[3]:.2f}'
            print(info)
            infer_info.append(info)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_pred_{single_metric[0]:.2f}.nii.gz")
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_scores.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + f"{ith}_gt.nii.gz")
        ith += 1

    final_metric = f'dice: {100*np.mean(total_metric["dice"]):.2f}({np.std(total_metric["dice"]):.2f}), jc: {100*np.mean(total_metric["jc"]):.2f}({np.std(total_metric["jc"]):.2f}), hd: {np.mean(total_metric["hd"]):.2f}({np.std(total_metric["hd"]):.2f}), asd: {np.mean(total_metric["asd"]):.2f}({np.std(total_metric["asd"]):.2f})'
    print(final_metric)
    infer_info.append(final_metric)
    
    with open(test_save_path+f'../{model_name}_performance_{nms}.txt', 'w') as f:
        for i in range(len(infer_info)):
            f.writelines(f'{infer_info[i]} \n')
    return final_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int32)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_single_case_average_output(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = net(test_patch)
                    num_outputs = len(y_logit)
                    y=torch.zeros(y_logit[0].shape).cuda()
                    for idx in range(num_outputs):
                        y += y_logit[idx]
                    y/=num_outputs
                    
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
