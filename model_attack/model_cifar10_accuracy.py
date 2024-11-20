import torch
from torch import nn
import numpy
import sys
import os
from prepare_models import model_cluster, model_list#引入模型集群
from prepare_datasets import train_loader, test_loader #引入数据加载器
from mora.models.ensemble_logits import Ensemble #引入mora的一个手段，用于输出模型预测结果
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ensembel_model = Ensemble(model_list).to(device)
ensembel_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = ensembel_model(images)
        final_output = outputs[-1]
        predicted_labels = final_output.max(1)
        print(f"Predicted_labels: {predicted_labels}")
        print(f"True_labels: {labels}")
        predicted_labels = predicted_labels[1]  # 获取最大值的索引
        predicted_labels = predicted_labels.to(device)
        labels = labels.to(device)
        correct_predictions = torch.eq(predicted_labels, labels).sum().item()
        accuracy = correct_predictions / len(labels)
        print(f"Accuracy in cifar10 for clean data of this batch: {accuracy * 100:.2f}%")
