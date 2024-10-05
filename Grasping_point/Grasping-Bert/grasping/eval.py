import os
import math
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from grasping.modules.dataloader import GraspDataset
from grasping.modules.model import Grasp_Bert
from grasping.args import Arguments

device = 'cuda' if torch.cuda.is_available() else 'cpu'