CUDA_VISIBLE_DEVICES=4,3 python run_eval.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--batch_size 100 --control "gender" --checkpoint 5000