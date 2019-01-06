# Implementation of "Image Style Transfer Using Convolutional Neural Networks, by Gatys" paper
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from tqdm import trange

STEPS = 300

device = torch.device('cuda')

layer_id_to_name = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1',
}


def get_model():
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # freeze all VGG parameters
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def extract_features(x, model):
    features = {}
    for layer_id, layer in model._modules.items():
        x = layer(x)
        if layer_id in layer_id_to_name:
            features[layer_id_to_name[layer_id]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(d * h * w)


def load_image(img_path, size=None):
    image = Image.open(img_path).convert('RGB')
    if size:
        image = image.resize(size)

    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # add the batch dimension
    image = in_transform(image).unsqueeze(0)
    return image


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def run(content_path, style_path, beta=1e6):
    print(f"[+] styling with beta={beta}")
    model = get_model()

    content = load_image(content_path).to(device)
    style = load_image(style_path, size=content.shape[-2:]).to(device)

    content_features = extract_features(content, model)
    style_features = extract_features(style, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1,
                     'conv2_1': 1,
                     'conv3_1': 1,
                     'conv4_1': 1,
                     'conv5_1': 1}

    alpha = 1  # alpha

    optimizer = optim.LBFGS([target])

    pbar = trange(STEPS)
    for _ in pbar:
        def closure():
            optimizer.zero_grad()

            target_features = extract_features(target, model)
            content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])

            style_loss = 0
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                style_loss += style_weights[layer] * F.mse_loss(target_gram, style_gram)

            total_loss = alpha * content_loss + beta * style_loss

            total_loss.backward()
            pbar.set_description(f"loss: {total_loss.item():.1f}")
            return total_loss

        optimizer.step(closure)

    return im_convert(target)
