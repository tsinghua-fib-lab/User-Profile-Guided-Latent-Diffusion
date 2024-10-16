"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import pdb
from improved_diffusion.text_datasets import load_data_text
import pickle

import setproctitle
setproctitle.setproctitle("sample")


def main():
    set_seed(101)
    args = create_argparser().parse_args()
    th.cuda.set_device(args.gpu)
    print("setting gpu: ", args.gpu)

    eta = args.eta

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    training_args['diffusion_steps'] = args.new_diffusion_steps
    args.__dict__.update(training_args)
    args.sigma_small = True
    print("===========================================================================================")

    # genout path
    file_name = f"gen_data_step_{os.path.split(args.model_path)[1].split('.')[-2].split('_')[-1]}.pkl"
    temp = os.path.split(args.model_path)[0].split('/')
    config_path = os.path.join("genout",temp[1])
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    gen_path =os.path.join(config_path, file_name)

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    print("device=",dist_util.dev())
    model.eval() # DEBUG

    if args.experiment_mode == 'conditional_gen': # TODO
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        print('conditional generation mode --> load data')
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        # print(rev_tokenizer)
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split=args.split,
            load_vocab=rev_tokenizer,
        )

    logger.log("sampling...")
    all_images = []
    all_labels = []
    model3 = get_weights(model.word_embedding, args)
    word_lst_e2e = []

    iter_num = args.num_samples//args.batch_size
    for i in range(iter_num):
        model_kwargs = {}
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop) # False
        # sample_fn = diffusion.p_sample_loop
        # sample_fn = diffusion.ddim_sample_loop
        sample_shape = (args.batch_size, args.max_pos+1, args.in_channel) # (bs, 169, in_channel)
        sample_temp = []

        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            eta=eta,
            top_p =args.top_p,
        )# torch.Size([bs, 169, in_channel])

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        sample_temp.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {(i+1) * args.batch_size} samples")

        arr = np.concatenate(sample_temp, axis=0)
        x_t = th.tensor(arr).cuda()
        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices # torch.Size([bs, 168, 1])

        for seq in sample:
            tokens = seq.squeeze(-1)
            word_lst_e2e.append(tokens.cpu().numpy()[1:])


    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    dist.barrier()
    logger.log("sampling complete")
    logger.log('decode by rounding.')
    if diffusion.training_mode.startswith('e2e'):
        word_lst = np.array(word_lst_e2e)
    print("=================================================")
    pickle.dump(word_lst, open(gen_path,'wb'))
    print("well done")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        new_diffusion_steps=2000,
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp', gpu=0, eta=1.)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
