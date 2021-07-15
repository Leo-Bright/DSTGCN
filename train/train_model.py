import copy
import time
from typing import Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.load_config import get_attribute
from utils.metric import evaluate
from utils.util import convert_train_truth_to_gpu
from utils.util import save_model, convert_to_gpu


def train_model(model: nn.Module,
                data_loaders: Dict[str, DataLoader],
                loss_func: callable,
                optimizer,
                model_folder: str,
                tensorboard_folder: str):

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_folder)
    num_epochs = get_attribute('epochs')

    since = time.perf_counter()

    model = convert_to_gpu(model)
    loss_func = convert_to_gpu(loss_func)

    save_dict, best_f1_score = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)
    test_metric = None
    try:
        for epoch in range(num_epochs):

            running_loss, running_metrics = {phase: 0.0 for phase in phases}, {phase: dict() for phase in phases}
            save_validate_this_epoch = False
            for phase in phases:
                if phase == 'train':

                    continue

                    model.train()
                else:
                    model.eval()

                steps, predictions, targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(data_loaders[phase]))
                for step, (g, spatial_features, temporal_features, external_features, truth_data) in tqdm_loader:

                    if step < 160:
                        continue

                    if not get_attribute("use_spatial_features"):
                        torch.zero_(spatial_features)
                    if not get_attribute("use_temporal_features"):
                        torch.zero_(temporal_features)
                    if not get_attribute("use_external_features"):
                        torch.zero_(external_features)

                    features, truth_data = convert_train_truth_to_gpu(
                        [spatial_features, temporal_features, external_features], truth_data)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(g, *features)
                        outputs = torch.squeeze(outputs)  # squeeze [batch-size, 1] to [batch-size]
                        try:
                            loss = loss_func(truth=truth_data, predict=outputs)
                        except:
                            print('=======loss=======')
                            print(truth_data)
                            print('=======loss=======')
                            print(list(truth_data.size()))
                            print('=======loss=======')
                            print(outputs)
                            print('=======loss=======')
                            print(list(outputs.size()))
                            print('=======loss=======')
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    targets.append(truth_data.cpu().numpy())
                    with torch.no_grad():
                        predictions.append(outputs.cpu().detach().numpy())

                    running_loss[phase] += loss * truth_data.size(0)
                    steps += truth_data.size(0)

                    tqdm_loader.set_description(
                        f'{phase:8} epoch: {epoch:3}, {phase:8} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                print(f'{phase} metric ...')
                try:
                    _cp = np.concatenate(predictions)
                    _ct = np.concatenate(targets)
                    scores = evaluate(_cp, _ct)
                except:
                    print('======scores========')
                    print('======predictions========')
                    print(predictions)
                    for idx, pred in enumerate(predictions):
                        print(str(idx), pred.size())
                    print('======targets========')
                    print(targets)
                    for idx, targ in enumerate(targets):
                        print(str(idx), targ.size())
                    print('======scores========')
                    exit(0)
                running_metrics[phase] = scores
                print(scores)

                if phase == 'validate' and scores['F1-SCORE'] > best_f1_score:
                    best_f1_score = scores['F1-SCORE']
                    save_validate_this_epoch = True
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                     epoch=epoch,
                                     optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                    print(f"save model as {model_folder}/model_{epoch}.pkl")
                    save_model(f"{model_folder}/model_{epoch}.pkl", **save_dict)

            scheduler.step(running_loss['train'])

            if save_validate_this_epoch:
                test_metric = running_metrics["test"].copy()

            for metric in running_metrics['train'].keys():
                writer.add_scalars(metric, {
                    f'{phase} {metric}': running_metrics[phase][metric] for phase in phases},
                                   global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(data_loaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    finally:
        print('======scores========')
        print('======predictions========')
        print(predictions)
        for idx, pred in enumerate(predictions):
            print(str(idx), pred.size())
        print('======targets========')
        print(targets)
        for idx, targ in enumerate(targets):
            print(str(idx), targ.size())
        print('======scores========')

        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        save_model(f"{model_folder}/best_model.pkl", **save_dict)

    return test_metric
