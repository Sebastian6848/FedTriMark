import torch

def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(labels.device)
    model = model.to(images.device)
    model.eval()
    images.requires_grad = True

    outputs = model(images)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.sign()
    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()
