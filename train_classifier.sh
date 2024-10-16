CUDA_VISIBLE_DEVICES=0,1,2,3 python train_run.py --experiment  e2e-back \
--app "--init_emb /data2/songyiwen/workspace/user_profile/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 150  --bsz 8 \
--dataset_name "mobile" --exp_n "gender"