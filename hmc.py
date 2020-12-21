import torch


def hamiltonian(x, v, model):
    energy = 0.5 * torch.pow(v, 2).sum(dim=1).sum(dim=1).sum(dim=1) + model(x).squeeze()
    return energy


def leapfrog_step(x, v, model, step_size, num_steps, label, sample=False):
    x = torch.log(x / (1 - x + 1e-10))
    x.requires_grad_(requires_grad=True)
    energy = model(torch.sigmoid(x))
    im_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * step_size * im_grad
    # x = x.detach()
    im_negs = []

    for i in range(num_steps):
        x.requires_grad_(requires_grad=True)
        energy = model(torch.sigmoid(x))

        im_grad = torch.autograd.grad([energy.sum()], [x])[0]
        v = v - step_size * im_grad
        x = x + step_size * v
        x = x.detach()
        v = v.detach()

        # if i % 10 == 0:
        #     print(i, hamiltonian(torch.sigmoid(x), v, model, label).mean(), torch.abs(im_grad).mean())


    x.requires_grad_(requires_grad=True)
    energy = model(torch.sigmoid(x))
    im_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * im_grad
    x = torch.sigmoid(x.detach())

    return x, v, im_grad


def gen_hmc_image(im_neg, step_size, temperature, model_fn, num_steps=10, sample=False):
    # energy = model.forward(im_neg, label)
    v = 0.1 * torch.randn_like(im_neg)

    im_neg_new, v_new, im_grad = leapfrog_step(im_neg, v, model_fn, step_size, num_steps, None)

    orig = hamiltonian(im_neg, v, model_fn)
    new = hamiltonian(im_neg_new, v_new, model_fn)

    mask = (torch.exp((orig - new)) > (torch.rand(new.size(0))).to(im_neg.device))
    im_neg_new[mask]= im_neg[mask]
    return im_neg_new
