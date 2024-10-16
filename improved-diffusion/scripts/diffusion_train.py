import argparse
import json, torch, os
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models
from myDataset import myDataset
from config import *
import pdb
from torch.utils.tensorboard import SummaryWriter
import setproctitle


def main():
    args = create_argparser().parse_args()
    #pdb.set_trace()
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)
    setproctitle.setproctitle(f"diffusion_exp{args.exp_n}@songyiwen")
    torch.cuda.set_device(args.gpu)
    print("setting gpu:", args.gpu)

    dist_util.setup_dist()
    logger.configure()
    writer = SummaryWriter(f"../diffusion_models/tensorboard_logs/exp_{args.exp_n}")

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) # 'uniform'

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # load data
    batch_size = args.batch_size
    splits = ['train', 'val', 'test']
    dataset = {}
    dataloader = {}
    for i in splits:
        dataset[i] = myDataset(args, split=i)
        dataloader[i] = torch.utils.data.DataLoader(dataset=dataset[i], 
                                            batch_size=batch_size,
                                            shuffle=True)

    def get_mapping_func(args, diffusion):
        model2 = load_models(args.in_channel, args.checkpoint_path, extra_args=args) 
        model3 = get_weights(model2, args)
        mapping_func = partial(compute_logp, args, model3.cuda())
        diffusion.mapping_func = mapping_func
        return mapping_func

    get_mapping_func(args, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader['train'],
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=dataloader['val'],
        eval_interval=args.eval_interval,
        writer=writer,
        layer_num=12
    ).run_loop()


if __name__ == "__main__":
    main()
