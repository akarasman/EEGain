import os
import copy

# os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm
import config
import click
import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV, AMIGOS, DREAMER, Seed, Dummy
from eegain.data.datasets_features import DREAMER_feat, DEAP_feat, SEED_feat, SEED_IV_feat
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception
from eegain.models.mlp import MLP
from collections import defaultdict
from dataclasses import asdict

from sklearn.metrics import *
from helpers import main_loso, main_loto, main_loso_fixed
from config import *


MAHNOB_transform = [
            eegain.transforms.Crop(t_min=30, t_max=-30),
            eegain.transforms.DropChannels(
                [
                    "EXG1",
                    "EXG2",
                    "EXG3",
                    "EXG4",
                    "EXG5",
                    "EXG6",
                    "EXG7",
                    "EXG8",
                    "GSR1",
                    "GSR2",
                    "Erg1",
                    "Erg2",
                    "Resp",
                    "Temp",
                    "Status",
                ]
            ),
            eegain.transforms.Filter(l_freq=0.3, h_freq=45),
            eegain.transforms.NotchFilter(freq=50),
            eegain.transforms.Resample(s_rate=128),
        ]

DEAP_transform = [
        eegain.transforms.DropChannels(
            [
                "EXG1",
                "EXG2",
                "EXG3",
                "EXG4",
                "GSR1",
                "Plet",
                "Resp",
                "Temp",
            ]
        ),
    ]

AMIGOS_transform = [
        eegain.transforms.DropChannels(
            [
            "ECG_Right",
            "ECG_Left",
            "GSR"
            ]
        ),
    ]

DREAMER_transform = [
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
    ]

SeedIV_transform = [
        # eegain.transforms.DropChannels(channels_to_drop_seed_iv),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
    ]

Seed_transform =  [
        # eegain.transforms.DropChannels(channels_to_drop_seed),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
    ]

Dummy_transform = [
        # Dummy transform
    ]

DREAMER_feat_transform = [
    
]

DEAP_feat_transform = [
    
]

SEED_feat_transform = [
    
]

SEED_IV_feat_transform = [
    
]

def generate_options():
    def decorator(func):
        config_instances = [ DREAMER_featConfig, DEAP_featConfig, SEED_featConfig, SEED_IV_featConfig, 
                             DummyConfig, TransformConfig, MAHNOBConfig,
                             TrainingConfig, EEGNetConfig, TSceptionConfig, DeepConvNetConfig,
                             ShallowConvNetConfig, MLPConfig,  ]
        for config_instance in config_instances:
            for field, value in asdict(config_instance()).items():
                option = click.option(f"--{field}", default=value, required=False, type=type(value))
                func = option(func)
        return func
    return decorator


@click.command()
@click.option("--model_name", required=True, type=str, help="name of the config")
@click.option("--data_name", required=True, type=str, help="name of the config")
@generate_options()
def main(**kwargs):
    transform = globals().get(kwargs["data_name"] + "_transform")
    transform.append(eegain.transforms.Segment(duration=kwargs["window"], overlap=0))
    transform = eegain.transforms.Construct(transform)
    dataset = globals()[kwargs['data_name']](transform=transform, root=kwargs["data_path"], **kwargs)

    # -------------- Model --------------
    if kwargs["model_name"]=="RANDOM":
        print("initializing random model")
        model = None
        empty_model = None
    # -------------- Model --------------
    elif kwargs["model_name"] == 'MLP':
        model = MLP(**kwargs)
        empty_model = copy.deepcopy(model)
    else:
        
        model = globals()[kwargs['model_name']](input_size=[1, kwargs["channels"], kwargs["window"]*kwargs["s_rate"]], **kwargs)
        empty_model = copy.deepcopy(model)
    if kwargs["split_type"] == "LOSO":
        classes = [i for i in range(kwargs["num_classes"])]
        main_loso(dataset, model, empty_model, classes, **kwargs)
    elif kwargs["split_type"] == "LOSO_Fixed":
        classes = [i for i in range(kwargs["num_classes"])]
        main_loso_fixed(dataset, model, empty_model, classes, **kwargs)
    else:
        classes = [i for i in range(kwargs["num_classes"])]
        main_loto(dataset, model, empty_model, classes, **kwargs)

if __name__ == "__main__":
    main()
