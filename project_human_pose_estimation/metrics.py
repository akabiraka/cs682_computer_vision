from models.deeppose import Deeppose
from datasets.mpii import MPII
import torch.optim as optim
from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# from progress.bar import Bar

import utility
# from metrics import PCKh
import torch
import numpy as np
import datasets.mpii_config as config

eps = 1e-8


class PCKh(object):
    """docstring for PCKh."""

    def __init__(self):
        super(PCKh, self).__init__()
        self.LB = -0.5 + eps if config.target_type == 'direct' else 0 + eps

    def calc_pckh(self, predictions, target, meta1, meta2, alpha=0.5):
        batchSize = predictions.shape[0]
        numJoints = 0 #-150
        numCorrect = 0
        for i in range(batchSize):
            index1 = 0
            index2 = 0
            skip = 0
            while (np.isnan(meta1[i, index1, :]).any() or (target[i, index1, :] <= self.LB).any()):
                index1 += 1
                if index1 >= 15:
                    skip = 1
                    break
            if skip:
                continue
            index2 = index1 + 1
            while (np.isnan(meta1[i, index2, :]).any() or (meta1[i, index2, :] == meta1[i, index1, :]).all() or (target[i, index2, :] <= self.LB).any() or (target[i, index2, :] == target[i, index1, :]).all()):
                index2 += 1
                if index2 >= 16:
                    skip = 1
                    break
            if skip:
                continue

            # Found 2 non-nan indices

            loaderDist = np.linalg.norm(
                target[i, index1, :] - target[i, index2, :])
            globalDist = np.linalg.norm(
                meta1[i, index1, :] - meta1[i, index2, :])
            effectiveHeadSize = meta2[i, 0] * (loaderDist / globalDist)

            for j in range(16):
                if j == 7 or j == 6:
                    continue
                if target[i, j, 0] >= self.LB and target[i, j, 1] >= self.LB and not(np.isnan(meta1[i, j, :]).any()):
                    numJoints += 1
                    if np.linalg.norm(predictions[i, j, :] - target[i, j, :]) <= alpha * effectiveHeadSize:
                        numCorrect += 1
        if numJoints == 0:
            return 1, 0
        return float(numCorrect) / float(numJoints), numJoints, numCorrect

    def calc_for_deeppose(self, out, target, meta1, meta2, alpha=0.5):
        out = out.reshape(-1, 16, 2).detach().cpu().numpy()
        target = target.reshape(-1, 16, 2).detach().cpu().numpy()
        pckh = self.calc_pckh(
            out, target, meta1.cpu().numpy(), meta2.cpu().numpy(), alpha)
        return pckh

#
# pckh = PCKh()
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Deeppose()
# model.to(device)
#
# val_dataset = MPII(split='val')
# val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)
#
# models = [23, 24, 30, 31, 32]
# bar = Bar('Processing:', max=len(models))
# for model_no in models:
#     path = "output_models/best_model_{}.pth".format(model_no)
#     model.load_state_dict(torch.load(path))
#     model.eval()
#
#     with torch.no_grad():
#         mpii = val_dataset
#         # using per sample image
#     #     for i in tqdm(range(10)):
#     #         inp, target, meta1, meta2, inp_without_clip = mpii.__getitem__(i)
#     #         # print(inp.shape, target.shape, meta1.shape, meta2.shape)
#     #         # raw_img, raw_pts = mpii.get_raw(0)
#     #         # print(raw_img.shape, raw_pts)
#         # inp = inp.unsqueeze(0).to(device)
#         # target = target.unsqueeze(0).to(device)
#         # meta1 = meta1.unsqueeze(0).to(device)
#         # meta2 = meta2.unsqueeze(0).to(device)
#         # # print(inp.shape, target.shape, meta1.shape, meta2.shape)
#         # out = model(inp)
#         # avg_correct_joint, n_joints, ncorrect = pckh.calc_for_deeppose(
#         #     out, target, meta1, meta2)
#         # print("PCKh: {:.4f}, n_joints: {}, n_correct: {}".format(
#         #     avg_correct_joint, n_joints, ncorrect))
#
#         # using val_loader
#         values = []
#         bar1 = Bar('Processing:', max=len(val_loader))
#         for i, data in enumerate(val_loader):
#             inp, target, meta1, meta2, _ = data
#             inp = inp.to(device)
#             target = target.to(device)
#             out = model(inp)
#             avg_correct_joint, n_joints, ncorrect = pckh.calc_for_deeppose(
#                 out, target, meta1, meta2)
#             # print(i, avg_correct_joint, n_joints, ncorrect)
#             values.append(avg_correct_joint)
#             bar1.next()
#         bar1.finish()
#
#         print("PCKh for model {} is {}".format(model_no, np.mean(values)))
#
#     bar.next()
# bar.finish()
