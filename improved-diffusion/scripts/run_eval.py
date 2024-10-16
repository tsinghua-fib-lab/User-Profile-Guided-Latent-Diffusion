"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2
from infill_util import langevin_fn3
from utils import merge,identify_home_
import pdb
import pickle
from run_clm import myDataset_mobile, myDataset_tencent
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

base_dir = '/data/zmy/'

def main():
    set_seed(101)
    args = create_argparser().parse_args()
    if_augment = args.if_augment
    if_filter = args.if_filter
    batch_size = args.batch_size
    exp_n = args.exp_n
    checkpoint = args.checkpoint
    control = args.control
    dataset_name = args.dataset
    
    print(control,'-'*100)
    assert(control!=None)

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)
    print("config_path=", config_path)

    args.noise_level = 0.0
    args.sigma_small = True
    args.diffusion_steps = 200
    args.batch_size = batch_size

    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()
    model_embs = th.nn.Embedding(training_args["num_grid"], training_args["in_channel"])
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)

    # load data
    splits = ['train', 'val', 'test']
    dataset = {}
    dataloader = {}
    for i in splits:
        if dataset_name == 'mobile':
            dataset[i] = myDataset_mobile(split=i,control=control,is_augment=if_augment)
        else:
            dataset[i] = myDataset_tencent(split=i,control=control,is_augment=if_augment)
        dataloader[i] = th.utils.data.DataLoader(dataset=dataset[i], 
                                            batch_size=batch_size,
                                            shuffle=True)

    # model
    if if_augment:
        print("augment",'-'*100)
        model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}_augment'
        # model_dir = f'/data/zmy/human_traj_diffusion/classifier_models/{control}_{args.dataset}'
    elif if_filter:
        print("filter",'-'*100)
        model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}_filter'
    else:
        model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}/checkpoint-{checkpoint}'
    print(model_dir)
    model_control = Classifier_GPT2.from_pretrained(model_dir).cuda()
    model_control.diffusion = diffusion
    model_control.eval()

    split = 'test'
    t_check = [10,50,100,190]
    len_ = len(t_check)
    data = [[] for i in range(len_)]
    pred = [[] for i in range(len_)]
    
    for step, inputs in enumerate(dataloader[split]):
        for id_, t in enumerate(t_check):
            output = model_control.eval_acc(**inputs, t=t)
            data[id_].extend(output['data'])
            pred[id_].extend(output['pred'])
    for i in range(len_):
        print(t_check[i],':')
        acc = round(accuracy_score(data[i], pred[i]), 5)
        f1_we = round(f1_score(data[i], pred[i], average='weighted'), 5)
        f1_ma = round(f1_score(data[i], pred[i], average='macro'), 5)
        print(acc,f1_we,f1_ma)
        print("-"*50)




def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0, batch_size=1, dataset='mobile', gpu=0, exp_n=0, file_name=0, checkpoint=1000,
        if_augment=False, control=None, if_filter=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = main()