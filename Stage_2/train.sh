exp_name="Stage_2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python vadepthnet/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log