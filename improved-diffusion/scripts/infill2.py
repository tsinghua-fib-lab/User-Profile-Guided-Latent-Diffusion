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

import setproctitle
setproctitle.setproctitle("infill@songyiwen")

base_dir = '/data2/songyiwen/workspace/user_profile/'

def main():
    set_seed(101)
    args = create_argparser().parse_args()
    if_augment = args.if_augment
    if_filter = args.if_filter
    batch_size = args.batch_size
    exp_n = args.exp_n
    checkpoint = args.checkpoint
    control = args.control
    print(control,'-'*100)
    assert(control!=None)
    # th.cuda.set_device(args.gpu)

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
    model.load_state_dict(th.load(args.model_path,map_location=th.device("cuda" if th.cuda.is_available() else "cpu")))
    model.to(dist_util.dev())
    model.eval()
    model_embs = th.nn.Embedding(training_args["num_grid"], training_args["in_channel"])
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)


    # model
    if if_augment:
        print("augment",'-'*100)
        model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}_augment'
        # model_dir = f'/data/zmy/human_traj_diffusion/classifier_models/{control}_{args.dataset}'
    elif if_filter:
        print("filter",'-'*100)
        model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}_filter'
        
    else:
        if checkpoint != 0:
            model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}/checkpoint-{checkpoint}'
        else:
            model_dir = base_dir + f'human_traj_diffusion/classifier_models/{control}_{args.dataset}'
    print(model_dir)
    model_control = Classifier_GPT2.from_pretrained(model_dir).cuda()

    # total_params1 = sum(p.numel() for p in model.parameters())
    # total_params2 = sum(p.numel() for p in model_control.parameters())
    # print(total_params1,total_params2)

    
    # 控制label文件
    control_label_lst = [] # TODO
    if if_augment and control=='home':
        control_label_path = base_dir + f'dataset/control_target/ctrl_{control}_{args.dataset}_val_test.json'
    else:
        control_label_path = base_dir + f'dataset/control_target/ctrl_{control}_{args.dataset}.json'
    with open(control_label_path, 'r') as f:
        for line in f:
            control_label_lst.append(json.loads(line))
    
    # augment
    # if if_augment:
    #     i= args.file_name
    #     print(i*100,min((i+1)*100,len(control_label_lst)))
    #     control_label_lst = control_label_lst[i*100:min((i+1)*100,len(control_label_lst))]
    #     print(control_label_lst)
    #     print("===")

    control_constraints = []
    for label_class in control_label_lst:
        label = [-100] * 169 + [int(x) for x in label_class] #带上start token
        label_ids = th.tensor(label).unsqueeze(0) # torch.Size([1, 170])
        debug_lst = [args.k, args.coef]
        langevin_fn_selected = partial(langevin_fn3, debug_lst, model_control, model3.cuda(),
                                        label_ids.expand(args.batch_size, -1), 0.1)
        control_constraints.append((langevin_fn_selected, label_class))
    partial_seq = control_constraints
    print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-'*20)
        
    logger.log("sampling...")
    sample_dict = {}
    for control_helper in partial_seq:
        all_images = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            seqlen = args.max_pos + 1
            sample_shape = (args.batch_size, seqlen, args.in_channel, )
            langevin_fn_selected, label_class_attributes = control_helper
            print(label_class_attributes, '-*'*200)
            loop_func_ = (diffusion.p_sample_loop_progressive if not args.use_ddim else diffusion.ddim_sample_loop_progressive)

            for sample in loop_func_(
                    model,
                    sample_shape,
                    denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                    # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                    #                     label_ids.expand(args.batch_size, -1), 0.1),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=next(model3.parameters()).device, # 
                    langevin_fn=langevin_fn_selected,
                    eta=args.eta,
            ):
                final = sample["sample"]
            
            sample = final
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample) 
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples]) # 1,bs,169,in_channel
            logger.log(f"created {len(all_images) * args.batch_size} samples")
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples] # emb
        sample_dict[str(label_class_attributes[0])] = arr

    dist.barrier()
    logger.log("sampling complete")

    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)

        for k, v in sample_dict.items():
            arr = v
            word_lst_e2e = []
            x_t = th.tensor(arr).cuda()
            reshaped_x_t = x_t
            logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            sample = cands.indices
            for seq in sample:
                tokens = seq.squeeze(-1)
                word_lst_e2e.append(list(tokens.cpu().numpy()[1:]))
            # word_lst = np.array(word_lst_e2e)
            result_dict[k] = word_lst_e2e
        return result_dict

    # decode
    print(f'sampled for {len(sample_dict)} control tasks')
    # exp name 生成结果
    if if_augment:
        out_path_pipe = os.path.join(base_dir,'human_traj_diffusion/improved-diffusion/genout_control',
                                    f"{control}_control_{args.dataset}_{exp_n}_{args.file_name}.json")
    else:
        if checkpoint != 0:
            out_path_pipe = os.path.join(base_dir,'human_traj_diffusion/improved-diffusion/genout_control',
                                        f"{control}_control_{args.dataset}_{exp_n}_{checkpoint}.json")
        else:
            out_path_pipe = os.path.join(base_dir,'human_traj_diffusion/improved-diffusion/genout_control',
                                        f"{control}_control_{args.dataset}_{exp_n}.json")

    result_dict = decode_helper(args, sample_dict, diff_model=model)
    with open(out_path_pipe, 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False,default=default_dump)

    print(f'written the decoded output to {out_path_pipe}')
    args.out_path2 = out_path_pipe
    return args

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
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0, batch_size=1, dataset='mobile', gpu=0, exp_n=0, k=3, coef=0.01, file_name=0,
        if_augment=False,control=None,if_filter=False,checkpoint=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def eval():
    set_seed(101)
    args = create_argparser().parse_args()
    control = args.control
    dataset = args.dataset
    exp_n = args.exp_n

    # for home
    from collections import Counter
    file_path = base_dir + 'human_traj_diffusion/improved-diffusion/genout_control/'
    # accuracy
    with open(file_path + f'{control}_control_{dataset}_{exp_n}.json' ,'r', encoding = 'utf-8') as fp:
        data = json.load(fp)
    
    num_sample = 30 # TODO
    scores = []
    word_lst = []
    # 生成数量采样 mobile
    # home_num_path = f'/data/zmy/dataset/control_target/home_{dataset}_num_val_test.json' # TODO
    home_num_path = base_dir + f'dataset/control_target/{control}_{dataset}_num.json'

    with open(home_num_path,'r',encoding='utf-8') as f:
        home_num = json.load(f)
    # mobile采样
    for k,v in data.items():
        v = np.array(v)
        home_true = k
        score = 0
        homes = []
        for traj in v:
            traj_temp = merge(traj)
            home = identify_home_(traj_temp)
            homes.append(home)
            if abs(int(home)-int(home_true))<4:
                score += 1
        scores.append(score)
        homes = Counter(homes)
        print(home_true)
        print(homes.most_common(10))
        print("========")
    print("well done")
    print(scores)
    print(np.mean(np.array(scores)/num_sample))

    # 按比例生成用于eval的数据
    # nums = [v for v in home_num.values()]

    for k,v in data.items():
        v = np.array(v)
        v = np.concatenate([v,v],axis=0)
        home_true = k
        if home_true in home_num.keys():
            sample_num = home_num[home_true]
            cnt_num = 0
            for traj in v:
                if cnt_num < sample_num: # mobile
                    word_lst.append(traj)
                    cnt_num += 1

    word_lst = np.array(word_lst)
    print(word_lst.shape)
    print("=================================================")
    pickle.dump(word_lst, open(file_path + f'home_control_{dataset}_{exp_n}.pkl','wb'))
    print("well done")

def eval_augment():
    '''合并augment生成的数据'''
    base_dir = base_dir + 'human_traj_diffusion/improved-diffusion/genout_control/'
    home_gen_all = {}
    for i in range(4):
        file_fir = base_dir + f'home_control_mobile_8_{i}.json'
        with open(file_fir,'r',encoding='utf-8') as f:
            home_gen = json.load(f)
            home_gen_all.update(home_gen)
    print(len(home_gen_all))
    with open(base_dir + 'home_control_mobile_8.json', 'w') as f:
        json.dump(home_gen_all, f, ensure_ascii=False,default=default_dump)
    print("well done")

def label_pkl_generate():
    import random

    # json按轨迹条数转json_new
    # 以及用于jsd评估的pkl
    args = create_argparser().parse_args()
    # control = args.control
    # dataset = args.dataset
    # exp_n = args.exp_n
    control = 'home'
    dataset = 'mobile'
    exp_n = 0

    print(control,exp_n,'*'*200)
    from collections import Counter
    file_path = base_dir + 'human_traj_diffusion/improved-diffusion/genout_control/'
    # accuracy
    with open(file_path + f'{control}_control_{dataset}_{exp_n}.json' ,'r', encoding = 'utf-8') as fp:
        data = json.load(fp) # 生成轨迹
    
    for k,v in data.items():
        num_sample = len(v) # TODO
        break
    print("num_sample=",num_sample)
 
    word_lst = []
    # 生成数量 mobile
    home_num_path = base_dir + f'dataset/control_target/{control}_{dataset}_num.json' # TODO
    # home_num_path = f'/data/zmy/dataset/control_target/{control}_{dataset}_num_val_test.json' # TODO
    with open(home_num_path,'r',encoding='utf-8') as f:
        home_num = json.load(f)

    # mobile采样
    max_num = np.max(np.array(list(home_num.values())))
    iter = int(max_num/num_sample) + 1 #重复几遍数据
    
    for k,v in data.items():
        v = np.array(v)
        v = np.concatenate([v for j in range(iter)],axis=0)
        random.shuffle(list(v))
        v = np.array(v)
        v_new = []
        label_true = k
        if (label_true == '32392' and control == 'age'):
            label_true = '32393'
        sample_num = home_num[label_true] #要采多少条
        cnt_num = 0
        for traj in v:
            if cnt_num < sample_num: # mobile
                word_lst.append(traj)
                v_new.append(traj)
                cnt_num += 1
        data[k] = v_new

    # new dict, age等label可用classifier
    # with open(file_path + f'{control}_control_{dataset}_{exp_n}_new.json', 'w') as f:
    #     json.dump(data, f, ensure_ascii=False,default=default_dump)
    
    if control=='home':
        traj_num = np.sum(np.array(list(home_num.values())))
        # home score
        score = 0
        for k,v in data.items():
            print(len(v))
            label_true = k
            v = np.array(v)
            homes = []
            for traj in v:
                traj_temp = merge(traj)
                home = identify_home_(traj_temp)
                homes.append(home)
                if abs(int(home)-int(label_true))<4:
                    score += 1
            homes = Counter(homes)
            print(label_true)
            print(homes.most_common(10))
        final_score = score/traj_num
        print("home score=",final_score)

    # 用于JSD计算的pkl
    word_lst = np.array(word_lst)
    print(word_lst.shape)
    print("=================================================")
    pickle.dump(word_lst, open(file_path + f'{control}_control_{dataset}_{exp_n}.pkl','wb'))
    print("well done")
    

if __name__ == "__main__":
    args = main()
    # eval(control,dataset,exp_n)
    # eval_augment()
    # label_pkl_generate()