#!/usr/bin/env python
# coding=utf-8
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
import torch
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from custom_trainer import Classifier_GPT2
import pdb
import numpy as np

import json
from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import setproctitle
setproctitle.setproctitle(f"classifier@songyiwen")

import numpy as np
import json
import pickle
import torch
import pdb
import os
from utils import merge,identify_home_
check_min_version("4.17.0.dev0")
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


base_dir = '/data2/songyiwen/workspace/user_profile/'
# base_dir = '/root/Desktop/workspace/songyiwen/'

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    learned_emb: Optional[str] = field(
        default='no',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/ROCstory',
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='/u/scr/xlisali/diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default='/u/scr/xlisali/e2e_data',
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )


    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    exp_n: Optional[str] = field(default=None,)

    batch_size: Optional[int] = field(default=128)

    augment: Optional[bool] = field(
        default=0,
        metadata={"help": "If training from scratch, pass a model type from the list: "},
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default='mobile', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class myDataset_mobile(object):
    def __init__(self, split:str, control:str, is_augment:bool):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grid = 180
        self.max_pos = 168
        self.control = control
        print("control:",control)

        if is_augment:
            self.split_data_dir = base_dir + f'dataset/mobile/split_label_augment_{self.control}' # home增强实验
            print("augment",'-'*100)
        else:
            self.split_data_dir = base_dir + f'dataset/mobile/split_label'
        print(self.split_data_dir)
        self.read_split_data(split)

    def read_split_data(self,split:str):
        path = os.path.join(self.split_data_dir, split+'.pkl')
        self.split_data = pickle.load(open(path, 'rb'))
        print(split+" data loaded...")

    def __getitem__(self, id):
        item = np.zeros((self.max_pos+1,3), dtype=int) #最后一个给home
        idx = 1 # 0为start token
        traj_data = self.split_data[id]['traj'] #mobile label
        revenue,gender,edu,age = self.split_data[id]['profile'] #mobile
        
        '''traj信息'''
        for j in range(len(traj_data)): # 天数*24
            for i in range(24):
                if idx < self.max_pos + 1:
                    x = traj_data[j][i][0]
                    y = traj_data[j][i][1]
                    item[idx][0] = x * self.max_grid +y +1 # grid id
                    item[idx][1] = i +1 # hour
                    item[idx][2] = traj_data[j][i][2] +1 # y # week
                    idx += 1
                else:
                    break

        traj = item[:,0,...].copy() # traj
        hours = item[:,1,...].copy()
        weeks = item[:,2,...].copy()
        hours = np.append(hours,0)
        weeks = np.append(weeks,0)

        # profile信息
        if self.control == 'home':
            traj_temp = merge(traj[1:])
            home = identify_home_(traj_temp)
            traj = np.append(traj,int(home)) # 带label的traj (170,)
        
        elif self.control == 'revenue':
            if revenue<40:
                revenue_label = 1 # 32392
            elif revenue<80:
                revenue_label = 2
            elif revenue<110:
                revenue_label = 3
            else:
                revenue_label = 4
            revenue_label += 32391
            traj = np.append(traj,int(revenue_label))

        elif self.control == 'edu':
            if edu.startswith('初中'):
                edu_label = 1
            elif edu.startswith('高中'):
                edu_label = 2
            elif edu.startswith('本科'):
                edu_label = 3
            elif edu.startswith('研究生'):
                edu_label = 4
            else:
                raise ValueError("edu error")
            edu_label += 32391
            traj = np.append(traj,int(edu_label))

        elif self.control == 'age':
            if age<30:
                age_label = 1
            elif age<40:
                age_label = 2
            elif age<60:
                age_label = 3
            else:
                age_label = 4
            age_label += 32391
            traj = np.append(traj,int(age_label))

        elif self.control == 'gender':
            if int(gender)==1:
                gender_label = 1
            else:
                gender_label = 2
            gender_label += 32391
            traj = np.append(traj,int(gender_label))

        labels = traj.copy()
        labels[1:self.max_pos+1] = -100
        
        traj = torch.tensor(traj)
        traj = traj.to(self.device)
        hours = torch.tensor(hours)
        hours = hours.to(self.device)
        weeks = torch.tensor(weeks)
        weeks = weeks.to(self.device)
        labels = torch.tensor(labels)
        labels = labels.to(self.device)

        return {"input_ids":traj,
                "labels":labels,
                "input_hours":hours,
                "input_weeks":weeks}

    def __len__(self):
        return len(self.split_data)


class myDataset_tencent(object):
    def __init__(self, split:str, control:str,is_augment:bool):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grid = 180
        if is_augment:
            raise ValueError("tencent doesnt have augmentation exp")
        else:
            self.split_data_dir = base_dir + 'dataset/tencent/split_label'
        self.max_pos = 168
        self.control = control
        print("control:",control)
        self.read_split_data(split)

    def read_split_data(self,split:str):
        path = os.path.join(self.split_data_dir, split+'.pkl')
        self.split_data = pickle.load(open(path, 'rb'))
        print(split + " data loaded...")

    def __getitem__(self, id):
        item = np.zeros((self.max_pos+1,3), dtype=int) #最后一个给home
        idx = 1 # 0为start token
        traj_data = self.split_data[id]['traj'] #mobile label
        revenue,gender,edu,age,_ = self.split_data[id]['profile'] 
        
        '''traj信息'''
        for j in range(len(traj_data)): # 天数*24
            for i in range(24):
                if idx < self.max_pos + 1:
                    x = traj_data[j][i][0]
                    y = traj_data[j][i][1]
                    item[idx][0] = x * self.max_grid +y +1 # grid id
                    item[idx][1] = i +1 # hour
                    item[idx][2] = traj_data[j][i][2] +1 # y # week
                    idx += 1
                else:
                    break
        traj = item[:,0,...].copy() # traj
        hours = item[:,1,...].copy()
        weeks = item[:,2,...].copy()
        hours = np.append(hours,0)
        weeks = np.append(weeks,0)

        # profile信息
        if self.control == 'home':
            traj_temp = merge(traj[1:])
            home = identify_home_(traj_temp)
            traj = np.append(traj,int(home)) # 带label的traj (170,)
        
        elif self.control == 'revenue':
            if revenue==1 or revenue==2:
                revenue_label = 1
            elif revenue==3:
                revenue_label = 2
            else:
                revenue_label = 3
            revenue_label = revenue + 32391
            traj = np.append(traj,int(revenue_label))

        elif self.control == 'gender':
            gender_label = gender + 32391
            traj = np.append(traj,int(gender_label))

        elif self.control == 'edu':
            edu_ = 3
            if edu == 1 or edu == 2:
                edu_ = 1
            elif edu == 3:
                edu_ = 2
            else:
                edu_ = 3
            edu_label = edu_ + 32391
            traj = np.append(traj,int(edu_label))

        elif self.control == 'age':
            if age<25:
                age_label = 1
            elif age<30:
                age_label = 2
            elif age<40:
                age_label = 3
            else:
                age_label = 4
            age_label += 32391
            traj = np.append(traj,int(age_label))
        else:
            raise ValueError("control error!")

        labels = traj.copy()
        labels[1:self.max_pos+1] = -100
        
        traj = torch.tensor(traj)
        traj = traj.to(self.device)
        hours = torch.tensor(hours)
        hours = hours.to(self.device)
        weeks = torch.tensor(weeks)
        weeks = weeks.to(self.device)
        labels = torch.tensor(labels)
        labels = labels.to(self.device)
        
        return {"input_ids":traj,
                "labels":labels,
                "input_hours":hours,
                "input_weeks":weeks}

    def __len__(self):
        return len(self.split_data)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dataset_name = data_args.dataset_name # mobile,tencent
    control = model_args.exp_n # control类别
    is_augment = model_args.augment
    batch_size = model_args.batch_size

    print("control:",control)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.vocab_size = 32401
    config_path = os.path.join(model_args.init_emb, "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args2 = json.load(f)
    training_args2['sigma_small'] = True
    training_args2['diffusion_steps'] = 200

    # load data
    splits = ['train', 'val', 'test']
    dataset = {}
    dataloader = {}
    for i in splits:
        if dataset_name == 'mobile':
            dataset[i] = myDataset_mobile(split=i,control=control,is_augment=is_augment)
        else:
            dataset[i] = myDataset_tencent(split=i,control=control,is_augment=is_augment)
        dataloader[i] = torch.utils.data.DataLoader(dataset=dataset[i], 
                                            batch_size=batch_size,
                                            shuffle=True)


    # also loading the diffusion model.
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update(training_args2)
    _, diffusion = create_model_and_diffusion(
        **temp_dict
    )

    # config.input_emb_dim = model_args.n_embd
    config.input_emb_dim = training_args2['in_channel']
    config.train_diff_steps = training_args2['diffusion_steps'] # 200
    config.n_positions = 256
    config.n_embd = training_args2['hidden_size']

    # 定义model
    model = Classifier_GPT2(config=config, diffusion=diffusion,)
    # pdb.set_trace()

    
    filename = model_args.init_emb

    if dataset_name == 'mobile':
        path_learned = '{}/ema_0.9999_225000.pt'.format(filename) ## 
    elif dataset_name == 'tencent':
        path_learned = '{}/ema_0.9999_220000.pt'.format(filename) ## tencent
    else:
        raise ValueError("dataset should be mobile or tencent")

    print('loading the learned embeddings')
    learned_embeddings = torch.load(path_learned,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))['word_embedding.weight']
    
    model.transformer.wte.weight.data = learned_embeddings.clone()
    model.transformer.wte.weight.requires_grad = False

    print("training")
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataloader['train'], # dataloader
        eval_dataset=dataloader['test'],
        tokenizer=None,
        data_collator=default_data_collator,
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.save_model()
    print("===")

    ''' train '''
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    print("=========================")
    print("well done")
    
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(dataloader['train'])
    )
    metrics["train_samples"] = min(max_train_samples, len(dataloader['train']))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()