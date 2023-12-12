exp_name="1103_0_256_samples_70000_eval"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python vadepthnet/eval.py configs/arguments_eval_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log