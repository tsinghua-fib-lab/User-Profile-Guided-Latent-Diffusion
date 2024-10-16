# condition
# mobile
CUDA_VISIBLE_DEVICES=1,0 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 4 --k 5 --coef 0.001 --num_samples 50 --batch_size 50 --control "gender"