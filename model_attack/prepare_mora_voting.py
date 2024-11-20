from mora.mora_attack import MORAAttack
import torch
import torch.nn as nn
from mora.mora import MORA
from prepare_models import models #引入模型集群
from mora.mora_attack import MORAAttack
from mora.models.ensemble_voting import Ensemble #引入mora的一个手段，用于输出模型预测结果
from prepare_datasets import train_loader, test_loader #引入数据加载器
# model = models
# ensembel_model = Ensemble(models).to(device)

# ensembel_model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ensemble_model = Ensemble(models).to(device)
ensemble_model.eval()
attack = MORAAttack(
    model = ensemble_model,
    n_iter=100, 
    norm='Linf', 
    n_restarts=1,
    eps=None,
    seed=0, 
    loss='ce', 
    eot_iter=1, 
    rho=.75, 
    verbose=False,
    device='cuda', 
    decay_step='linear', 
    float_dis=1.0, 
    version='pgd', 
    ensemble_pattern='softmax'
)

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    break

acc, adv_data = attack.perturb(images, labels, scale=0.5)
#将生成的对抗样本输入到模型，验证其是否被成功误分类
with torch.no_grad():
    output = ensemble_model(adv_data)
    pred = output.argmax(dim=1)
    success_rate = (pred != labels).float().mean()
print("对抗成功率：{:.2%}".format(success_rate.item()))


#可视化对抗样本
import matplotlib.pyplot as plt

# 显示原始样本和对抗样本
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(images[0].cpu().squeeze().numpy(), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Adversarial")
plt.imshow(adv_data[0].cpu().squeeze().numpy(), cmap="gray")
plt.show()










