import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets.mpii import MPII
from models.deeppose import Deeppose
from models.losses import DeepposeLose

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-5]
for nth_run, init_lr in enumerate(lrs):
    print("\n\nStarting 2{}th run".format(nth_run))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Deeppose()
    model.to(device)
    criterion = DeepposeLose()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    batch_size = 30
    n_epochs = 40
    print_every = 1
    test_every = 4
    plot_every = 2

    print("device=", device)
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("init_lr=", init_lr)
    print(model)

    train_dataset = MPII(split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    inp, target, meta1, meta2, _ = train_dataset.__getitem__(0)
    print(inp.shape, target.shape, meta1.shape, meta2.shape)
    print("train_dataset_size=", train_dataset.__len__())
    print("train_loader_size=", len(train_loader))

    val_dataset = MPII(split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    inp, target, meta1, meta2, _ = val_dataset.__getitem__(0)
    print(inp.shape, target.shape, meta1.shape, meta2.shape)
    print("val_dataset_size=", val_dataset.__len__())
    print("val_loader_size=", len(val_loader))

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = Deeppose()
    # model.to(device)
    # # print(model)
    # inp, target, _, _ = train_dataset.__getitem__(0)
    # inp = inp.unsqueeze(0).to(device)
    # target = target.unsqueeze(0).to(device)
    # # inp.shape
    # out = model(inp)
    # # print(out.dtype, target.dtype, inp.dtype)
    # # print(out, target)
    # criterion = DeepposeLose()
    # criterion(out, target)

    def train():
        model.train()
        loss = 0.0
        losses = []
        n_train = len(train_loader)
        for i, data in enumerate(tqdm(train_loader)):
            inp, target, _, _, _ = data
            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            # if i == 10:
            #     break
        return torch.stack(losses).mean().item()

    # train()

    def test(data_loader):
        model.eval()
        loss = 0.0
        losses = []
        n_data = len(data_loader)
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                inp, target, _, _, _ = data
                inp = inp.to(device)
                target = target.to(device)
                out = model(inp)
                loss = criterion(out, target)
                losses.append(loss)
                # if i == 10:
                #     break
        return torch.stack(losses).mean().item()

    # test(val_loader)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, n_epochs + 1)):
        # train_loss = 0.0
        train_loss = train()
        train_losses.append(train_loss)

        if epoch % print_every == 0:
            print("epoch:{}/{}, train_loss: {:.5f}".format(epoch,
                                                           n_epochs + 1, train_loss))

        if epoch % test_every == 0:
            # val_loss = 0.0
            val_loss = test(val_loader)
            print("epoch:{}/{}, val_loss: {:.5f}".format(epoch, n_epochs + 1, val_loss))
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('Updating best test loss: {:.5f}'.format(best_val_loss))
                torch.save(model.state_dict(),
                           'output_models/best_model_2{}.pth'.format(nth_run))
        if epoch % plot_every == 0:
            pass
            # plt.plot(train_losses)
            # plt.plot(val_losses)
            # plt.show()

    print("train_losses:\n", train_losses)
    print("val_losses:\n", val_losses)
