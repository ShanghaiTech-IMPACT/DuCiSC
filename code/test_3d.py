import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='Pancreas', help='dataset_name')
parser.add_argument('--output_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='DuCiSC', help='exp_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = FLAGS.output_path
test_save_path = FLAGS.output_path + f"/{FLAGS.exp}_predictions/"

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    with open('./data/LA/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["./data/LA/2018LA_Seg_TrainingSet/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

elif FLAGS.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    with open('data/Pancreas/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]

elif FLAGS.dataset_name == "Nurves":
    patch_size = (112, 112, 160)
    with open('data/Nurves/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["data/Nurves/Nurves_h5/" + item.replace('\n', '') + ".h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    save_result = False
    net = net_factory(net_type=FLAGS.exp, in_chns=1, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.exp))
    # save_mode_path = "./pretrained_weigthts/DuCiSC_Pancreas_12.pth"
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.exp, 1, net, image_list, num_classes=num_classes,
                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                        save_result=save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas":
        avg_metric = test_all_case(FLAGS.exp, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                        save_result=save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Nurves":
        avg_metric = test_all_case(FLAGS.exp, 1, net, image_list, num_classes=num_classes,
                        patch_size=(160, 128, 112), stride_xy=16, stride_z=16,
                        save_result=save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
