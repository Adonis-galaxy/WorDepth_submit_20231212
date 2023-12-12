exp_name="compare_eval"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python vadepthnet/eval.py configs/arguments_eval_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log