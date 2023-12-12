import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from networks.stage_3 import Depth_Estimator
from networks.stage_1_2 import Stage_2_Model,Stage_1_Model

import cv2

parser = argparse.ArgumentParser(description='WorDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='WorDepth')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--prior_mean',                type=float, help='prior mean of depth', default=1.54)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path_stage_3',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--checkpoint_path_stage_1',   type=str,   default='')
parser.add_argument('--checkpoint_path_stage_2',   type=str,   default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path_stage_3, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')


def vis_img(img, vis_name):
    # Normalize the feature map to be in the range 0-255
    feature_map=img
    # Normalize the feature map to 0-255 range
    normalized_feature_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit image
    normalized_feature_map = np.uint8(normalized_feature_map)

    # Apply a colormap (e.g., COLORMAP_JET)
    colored_feature_map = cv2.applyColorMap(normalized_feature_map, cv2.COLORMAP_JET)

    # Save the image
    cv2.imwrite(os.path.join(args.log_directory, args.model_name)+"/"+vis_name+".png", colored_feature_map)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def online_eval(model,model_1,model_2, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']

            # ====== Stage 1 and 2 inference ======
            text_feature_list=None # [B,C]
            image_feature_list=None
            for i in range(len(eval_sample_batched['sample_path'])): # B=4
                sample_path=eval_sample_batched['sample_path'][i].split(' ')[0][:-4].replace("/","_")
t
                text_feature_path=args.data_path_eval+eval_sample_batched['sample_path'][i].split(' ')[0][:-4]+'.pt'
                image_feature_path=args.data_path_eval+eval_sample_batched['sample_path'][i].split(' ')[0][:-4]+'_img_feat.pt'
    
                text_feature=torch.load(text_feature_path, map_location=image.device)
                image_feature=torch.load(image_feature_path, map_location=image.device)

                if i==0:
                    text_feature_list=text_feature
                    image_feature_list=image_feature
                else:
                    text_feature_list=torch.cat((text_feature_list,text_feature),dim=0)
                    image_feature_list=torch.cat((image_feature_list,image_feature),dim=0)

            eps=model_2(image_feature_list)
            inter_feature, prior_depth = model_1(image, text_feature=text_feature_list,eps=eps)
            
            std=[1.5,2,2,3,3]
            eps =  torch.normal(0, std[0], size=eps.size())
            _, stage_1_depth_0 = model_1(image, text_feature=text_feature_list,eps=eps)

            eps = torch.normal(0, std[1], size=eps.size())
            _, stage_1_depth_1 = model_1(image, text_feature=text_feature_list,eps=eps)

            eps =  torch.normal(0, std[2], size=eps.size())
            _, stage_1_depth_2 = model_1(image, text_feature=text_feature_list,eps=eps)
            
            eps =  torch.normal(0, std[3], size=eps.size())
            _, stage_1_depth_3 = model_1(image, text_feature=text_feature_list,eps=eps)

            eps =  torch.normal(0, std[4], size=eps.size())
            _, stage_1_depth_4 = model_1(image, text_feature=text_feature_list,eps=eps)

            # ====== Stage 1 and 2 inference Done ======
            
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image=image, inter_feature=inter_feature,prior_depth=prior_depth)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            prior_depth = prior_depth.cpu().numpy().squeeze()


            # vis_img(img=pred_depth,vis_name=sample_path+"_pred")
            vis_img(img=gt_depth,vis_name=sample_path+"_gt")
            vis_img(img=prior_depth,vis_name=sample_path+"_prior")
            vis_img(img=stage_1_depth_0.cpu().numpy().squeeze(),vis_name=sample_path+"_stage1_0")
            vis_img(img=stage_1_depth_1.cpu().numpy().squeeze(),vis_name=sample_path+"_stage1_1")
            vis_img(img=stage_1_depth_2.cpu().numpy().squeeze(),vis_name=sample_path+"_stage1_2")
            vis_img(img=stage_1_depth_3.cpu().numpy().squeeze(),vis_name=sample_path+"_stage1_3")
            vis_img(img=stage_1_depth_4.cpu().numpy().squeeze(),vis_name=sample_path+"_stage1_4")
            



        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    model = Depth_Estimator(pretrained=args.pretrain,
                       max_depth=args.max_depth,
                       prior_mean=args.prior_mean,
                       img_size=(args.input_height, args.input_width))

    model = torch.nn.DataParallel(model)
    model.cuda()



    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)

    model_just_loaded = False
    if args.checkpoint_path_stage_3 != '':
        if os.path.isfile(args.checkpoint_path_stage_3):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path_stage_3))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path_stage_3)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                print(loc)
                checkpoint = torch.load(args.checkpoint_path_stage_3, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path_stage_3, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path_stage_3))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')






    # ====== Stage 1 and 2 ======
    model_1 = Stage_1_Model()
    model_2 = Stage_2_Model()

    model_1 = torch.nn.DataParallel(model_1)
    model_2 = torch.nn.DataParallel(model_2)
                
    model_1.eval()
    model_2.eval()

    model_1.cuda()
    model_2.cuda()

    checkpoint_1 = torch.load(args.checkpoint_path_stage_1)
    model_1.load_state_dict(checkpoint_1['model'])
    print("Load Stage 1 Model")

    checkpoint_2 = torch.load(args.checkpoint_path_stage_2)
    model_2.load_state_dict(checkpoint_2['model'])
    print("Load Stage 2 Model")
    # ====== Stage 1 and 2 Done ======

    # ===== Evaluation before training ======
    model.eval()
    with torch.no_grad():
        eval_measures = online_eval(model,model_1,model_2,dataloader_eval, gpu, ngpus_per_node)


    exit("Vis Finish~")
    


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    # command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    # os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)


    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
