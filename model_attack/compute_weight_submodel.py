import torch
from prepare_mora_attack import attack
from prepare_datasets import test_loaer
from prepare_models import model_list
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idx = 0
batchs = 0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    batchs = batchs + 1
    accuracy, adversarial_samples, adv_acc_result_batch = attack.perturb(images, labels)
    modellist_output = ensembel_model(adversarial_samples)
    final_output = modellist_output[-1]
    predicted_labels = final_output.max(1)
    for model in model_list:
        idx = idx + 1
        signal_model_outputs = model(adversarial_samples)
        weights = Ensemble.analyze_weights(outputs=signal_model_outputs, labels=predicted_labels)
        print(f"针对第{batchs}批次数据，第{idx}模型的权重分析是：{weights}")

