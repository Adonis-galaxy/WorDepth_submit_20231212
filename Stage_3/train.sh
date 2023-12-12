exp_name="Stage_3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python wordepth/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log