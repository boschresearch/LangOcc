# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from mmdet3d.models.lang_occ_modules import ClipReducer, ClipReducerAE, ClassSeparationLoss, ClassSeparationLossMSE, ClassSeparationLossMargin
from mmdet3d.models.lang_occ_modules.vocabulary import vocabulary
import os
import numpy as np

def train_clip_reducer(class_embeds, vocabulary_map, model_type, separator=None, reduced_size=128, verbose=False, separation_weight=0,
                       use_templates=False, epochs=100, device=torch.device('cpu')):

    # Build model
    if model_type == 'U':
        reducer = ClipReducer(class_embeds.shape[-1], reduced_size).to(device)
    elif model_type == 'AE':
        reducer = ClipReducerAE(class_embeds.shape[-1], reduced_size).to(device)
    else:
        assert False, "Please provide a valid model type."

    # Build optimizer
    optimizer = torch.optim.Adam(reducer.parameters(), lr=0.001)
    if separator == 'Cos':
        separation_regularizer = ClassSeparationLoss(vocabulary_map)
    elif separator == 'MSE':
        separation_regularizer = ClassSeparationLossMSE(vocabulary_map)
    elif separator == 'Margin':
        separation_regularizer = ClassSeparationLossMargin(vocabulary_map)
    else:
        separation_regularizer = None

    loss_weight = 10
    eps = 1e-4
    prev_loss = 10000

    class_embeds = class_embeds.to(device).float()
    # Training Loop
    eps_loss_count = 0
    for e in tqdm(range(epochs)):
        bottleneck, recon = reducer(class_embeds)
        loss = loss_weight * reducer.loss(recon, class_embeds) 

        if separation_regularizer is not None:
            loss = loss + separation_weight * separation_regularizer(bottleneck)

        if torch.isnan(loss):
            break                

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (loss - prev_loss).abs() < eps:
            eps_loss_count += 1
        else:
            eps_loss_count = 0

        if loss < eps or eps_loss_count>30:
            break
        else:
            prev_loss = loss

    print(f'Final Loss after epoch {e}: {round(loss.item(), 4)}')
    return reducer

def store_reducer(model, args, verbose=False):
    save_path = os.path.join(args.save_path,
        f'reducer_{args.version}_{args.separation_weight}_{args.separator}_{args.model}_{args.reduced_size}{"_templates" if args.use_templates else ""}.pth')
    if verbose:
        print(f"Saving Reducer to {save_path}...")
    torch.save(model.state_dict(), save_path)
    if verbose:
        print("Done.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train the clip reducer')
    parser.add_argument('--embeddings-root', type=str, default="./data/embeddings")
    parser.add_argument('--save-path', type=str, default="./ckpts")
    parser.add_argument('--version', type=int, default=1, help="Vocabulary version")
    parser.add_argument('--reduced-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--model', nargs='?', default='U', choices=['U', 'AE'])
    parser.add_argument('--separator', nargs='?', default='None', choices=['Cos', 'MSE', 'Margin', 'None'])
    parser.add_argument('--separation-weight', type=float, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-templates', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    os.makedirs(args.save_path, exist_ok=True)

    # Load data from file
    embeddings_path = os.path.join(args.embeddings_root, 'MaskCLIP')
    class_embeds = torch.tensor(np.load(os.path.join(embeddings_path, f'class_embeddings_v{args.version}{"_templates" if args.use_templates else ""}.npz'))['arr_0'])
    vocabulary_map = vocabulary[args.version][1]

    print("Starting to train Reducer...")
    reducer = train_clip_reducer(class_embeds, vocabulary_map, args.model, args.separator, args.reduced_size, epochs=args.epochs,
                                   separation_weight=args.separation_weight, use_templates=args.use_templates, device=device, verbose=args.verbose)
    print("Done.")
    
    # Store model ckpt
    store_reducer(reducer, args, verbose=True)