import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from networks.vadepthnet import VADepthNet, Stage_2_Model


parser = argparse.ArgumentParser(description='VADepthNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='vadepthnet')
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
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./models/')
parser.add_argument('--stage_2_checkpoint_path',           type=str,   help='stage 2 checkpoint', default='')
parser.add_argument('--stage_1_checkpoint_path',           type=str,   help='stage 1 checkpoint', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq_ckpt',                 type=int,   help='Checkpoint saving frequency in global steps', default=10000)
parser.add_argument('--save_freq_pred',                 type=int,   help='Pred Vis frequency in global steps', default=1000)


# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)



# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:2305')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)


# Structure                                                                
parser.add_argument('--hidden_dim',                 type=int,   default=256)
parser.add_argument('--std_reg',                 type=float,   help='reg term of std norm, 0 as no reg',default=1 )


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    model_1 = VADepthNet(pretrained=args.pretrain,
                       max_depth=args.max_depth,
                       prior_mean=args.prior_mean,
                       img_size=(args.input_height, args.input_width),
                       hidden_dim=args.hidden_dim,
                       std_reg=args.std_reg
                       )
    model_2 = Stage_2_Model()

    model_1 = torch.nn.DataParallel(model_1)
    model_2 = torch.nn.DataParallel(model_2)
                
    model_1.train()
    model_2.train()

    model_1.cuda()
    model_2.cuda()


    print("== Model Initialized")

    global_step = 1


    # Training parameters, only model_2
    # optimizer = torch.optim.Adam([{'params':model_1.module.parameters(),"lr":3e-5},
    #                             {'params':model_2.module.parameters()}],
    #                             lr=args.learning_rate)
    
    optimizer = torch.optim.Adam([{'params':model_2.module.parameters()}],
                                lr=args.learning_rate)

    #Loading CKPT
    checkpoint_1 = torch.load(args.stage_1_checkpoint_path)
    model_1.load_state_dict(checkpoint_1['model'])
    print("Load Stage 1 Model")
    
    if args.stage_2_checkpoint_path:
        print("Load Stage 2 CKPT")
        checkpoint_2 = torch.load(args.stage_2_checkpoint_path)
        model_2.load_state_dict(checkpoint_2['model'])




    dataloader = NewDataLoader(args, 'train')

    
    # Logging
    eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
    eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)


    start_time = time.time()
    duration = 0

    # end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate



    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    print("Total Steps:",num_total_steps)
    print("Save Frequency:",args.save_freq_ckpt)
    print("Pred Vis Frequency:",args.save_freq_pred)
    Training_Success=False
    print("Start Training!")


    # Training Process
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True)) # torch.Size([B, 3, 480, 640])
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            

            text_feature_list=None # [B,C]
            image_feature_list=None
            for i in range(len(sample_batched['sample_path'])): # B=4
                # 读取feature_path的feature，append

                text_feature_path=args.data_path+sample_batched['sample_path'][i].split(' ')[0][:-4]+'.pt'
                image_feature_path=args.data_path+sample_batched['sample_path'][i].split(' ')[0][:-4]+'_img_feat.pt'
    
                text_feature=torch.load(text_feature_path, map_location=image.device)
                image_feature=torch.load(image_feature_path, map_location=image.device)

                if i==0:
                    text_feature_list=text_feature
                    image_feature_list=image_feature
                else:
                    text_feature_list=torch.cat((text_feature_list,text_feature),dim=0)
                    image_feature_list=torch.cat((image_feature_list,image_feature),dim=0)


            
            eps=model_2(image_feature_list)
            depth_est, loss, std_norm, depth_loss, smooth_loss = model_1(image, gts=depth_gt, text_feature=text_feature_list,eps=eps)
            path_inter=sample_batched['sample_path'][i].split(' ')[0][:-4]



            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1.0)
            
            optimizer.step()

            if Training_Success==False:
                Training_Success=True
                print("Start Training Sucessfully: One Step!")

            # Change Lr
            # for param_group in optimizer.param_groups:
            #     current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            #     param_group['lr'] = current_lr

            # optimizer.step()


            duration += time.time() - before_op_time

            # Log
            if global_step % args.log_freq==0:
                eval_summary_writer.add_scalar("Train Loss", loss.item(), int(global_step))
                eval_summary_writer.add_scalar("STD Norm", std_norm.item(), int(global_step))
                eval_summary_writer.add_scalar("Depth Loss", depth_loss.item(), int(global_step))
                eval_summary_writer.add_scalar("Smooth Loss", smooth_loss.item(), int(global_step))



            # Save Checkpoitns by frequency, vis pred
            if (global_step >= args.save_freq_ckpt and global_step % args.save_freq_ckpt ==0) or (global_step==num_total_steps):
                # Save CKPT
                model_save_name = '/model_{}'.format(global_step)
                print('Saving model. Step:',global_step)
                checkpoint = {'global_step': global_step,
                                'model': model_2.state_dict()}
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)        

            # Vis Depth Pred
            if global_step % (args.save_freq_pred) ==0 or (global_step==1):
                print('Print Depth Pred. Step:',global_step)

                for i in range(len(sample_batched['sample_path'])): # B=4
                    path_inter=sample_batched['sample_path'][i].split(' ')[0][:-4]

                    vis_path="models/"+ args.model_name +"/"+path_inter.replace("/","_")+"_"+str(global_step)+".png"
                    pred_depth = depth_est[i].detach().cpu().numpy().squeeze(0) / 10 * 255
                    cv2.imwrite(vis_path, pred_depth)  

                    
                    gt_depth=depth_gt[i].detach().cpu().numpy().squeeze(0)/ 10 * 255
                    vis_gt_path="models/"+ args.model_name +"/"+path_inter.replace("/","_")+"_gt"+".png"
                    cv2.imwrite(vis_gt_path, gt_depth)      

            model_just_loaded = False
            global_step += 1

            
        epoch += 1
       


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1


    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)
    os.system('cp ' + "vadepthnet/train.py" + ' ' + args_out_path + "/train.py.backup")
    os.system('cp ' + "vadepthnet/networks/vadepthnet.py" + ' ' + args_out_path + "/vadepthnet.py.backup")

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1


    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
