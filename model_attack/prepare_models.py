import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import DiffusionClassifier
import DiffusionClassifier.defenses.PurificationDefenses.DiffPure.DiffusionClassifier.EDMDC
from DiffusionClassifier.defenses.PurificationDefenses.DiffPure.DiffusionClassifier.EDMDC import EDMEulerIntegralDC
from DiffusionClassifier.models.unets.EDM.get_edm_nets import get_edm_cifar_cond
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelCluster:
    def __init__(self, num_models, timesteps_ranges, unet_loader):
        self.models = []
        for i in range(num_models):
            # 使用 get_edm_cifar_cond 加载预训练的 UNet 模型
            unet_model = unet_loader()
            # 分配给每个模型一个特定的时间步范围
            timesteps = timesteps_ranges[i]
            model = EDMEulerIntegralDC(unet=unet_model, timesteps=timesteps)
            self.models.append(model)

    def forward(self, x):
        outputs = [model.one_step_denoise(x, sigma=timesteps.mean()) for model, timesteps in zip(self.models, timesteps_ranges)]
        return torch.stack(outputs).mean(dim=0)

# 设置每个模型的时间步范围
num_models = 5  # 模型数量
full_timesteps = torch.linspace(1e-4, 3, 1000)  # 总时间步范围
timesteps_ranges = torch.chunk(full_timesteps, num_models)  # 将时间步划分给每个模型

# 创建模型集群
model_cluster = ModelCluster(
    num_models=num_models,
    timesteps_ranges=timesteps_ranges,
    unet_loader=lambda: get_edm_cifar_cond(pretrained=True)  # 加载每个模型的预训练 UNet
)








def create_model_list(num_models=5):
    # 定义总的时间步范围
    total_timesteps = torch.linspace(1e-4, 3, 1000)
    
    # 将 total_timesteps 等分成 num_models 份
    timestep_ranges = torch.split(total_timesteps, len(total_timesteps) // num_models)

    # 为每个子模型实例创建对应的时间步范围
    model_list = []
    for i, timesteps in enumerate(timestep_ranges):
        # 加载预训练的 UNet 模型
        unet_model = get_edm_cifar_cond(pretrained=True)
        
        diffusion_model = EDMEulerIntegralDC(unet=unet_model, timesteps=timesteps)
        model_list.append(diffusion_model)
    return model_list

model_list = create_model_list(num_models=5)

