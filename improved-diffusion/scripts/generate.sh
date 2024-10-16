python scripts/text_sample.py --model_path diffusion_models/exp58/ema_0.9999_170000.pt \
--num_samples 1000 --top_p 1.0 --out_dir genout --eta 1. --use_ddim True --gpu 0 \
--new_diffusion_steps 200 --batch_size 180