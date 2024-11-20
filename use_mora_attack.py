import torch
import torchvision as tv
from defenses.PurificationDefenses.DiffPure import (
    EDMEulerIntegralDC, DiffusionClassifierSingleHeadBaseWraped)
from models.unets.EDM.get_edm_nets import get_edm_cifar_cond
from mora.mora_attack import MORAAttack
from mora.models.ensemble_logits import Ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = tv.datasets.CIFAR10(
    root='../data/', train=False, download=True,
    transform=tv.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=2)

def create_model_list(num_models=5):
    total_timesteps = torch.linspace(1e-4, 3, 1000)
    timestep_ranges = torch.split(total_timesteps, len(total_timesteps) // num_models)
    model_list = []
    unet_model = get_edm_cifar_cond(pretrained=True)
    for i, timesteps in enumerate(timestep_ranges):
        # 加载预训练的 UNet 模型
        diffusion_model = EDMEulerIntegralDC(unet=unet_model, timesteps=timesteps)
        wrapped_model = DiffusionClassifierSingleHeadBaseWraped(diffusion_model)
        model_list.append(wrapped_model)
    return model_list


model_list = create_model_list(num_models=5)
models_ensemble = Ensemble(models=model_list)
model_attack = MORAAttack(
    model=models_ensemble,
    n_iter=100, norm='Linf', n_restarts=1, eps=8/255, seed=0,
    loss='ce', eot_iter=1, rho=.75, verbose=True, device='cuda', decay_step='linear',
    float_dis=1.0, version='pgd', ensemble_pattern='logits')

for batch, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    accuracy, adversarial_samples, adv_acc_result_batch = model_attack.perturb(images, labels)
    # submodel_weight_list = model_attack.get_sub_weight(x_ensemble=models_ensemble, )
    print(f"在第{batch}批次中,攻击模型集群的准确率：,{accuracy.float().mean().item()}")
    print(f"在第{batch}批次中，adv_acc_result_batch是:{adv_acc_result_batch}")

