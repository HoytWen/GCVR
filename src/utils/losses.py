import torch as th
from torch.nn.functional import softplus
import numpy as np
import torch.nn.functional as F

def estimatation_likelihood(x_resc, mu, logvar, target):

    likelihood = (-(mu - target)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    return likelihood

def estimatation_bound(mu, logvar, target, device):

    positive = - (mu - target) ** 2 / 2. / logvar.exp()
    prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
    y_samples_1 = target.unsqueeze(0)
    negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
    bound = F.relu((positive.sum(dim = -1) - negative.sum(dim = -1)).mean())

    return bound

def discriminator_loss(loss_func, real_logits, fake_logits, device):

    discriminator_real = loss_func(real_logits, th.ones(real_logits.shape[0]).long().to(device))
    discriminator_fake = loss_func(fake_logits, th.zeros(fake_logits.shape[0]).long().to(device))
    discriminator_loss = discriminator_real + discriminator_fake

    return discriminator_loss

def generator_loss(loss_func, fake_logits, device):

    generator_loss = loss_func(fake_logits, th.ones(fake_logits.shape[0]).long().to(device))

    return generator_loss


def normalized_mse_loss(h1, h2):

    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    loss = F.mse_loss(h1, h2)

    return loss

def normalized_l1_loss(h1, h2):

    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    loss = F.l1_loss(h1, h2)

    return loss