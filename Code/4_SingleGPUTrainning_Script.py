import transformers as trsf

import os
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import Tensor
import torch.multiprocessing as mp

#from datasets import Dataset

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


from sklearn.model_selection import train_test_split

import ModelsUtils as Utils
import Configurations as Configs
#import wsdm_modelutils as Utils

import peft as pft

import argparse
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch

#----------------------------------------------------------------------------------------
def evaluate_model(model, dataloader, device=0):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    loss_fn = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), unit='row'):

            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                features=batch['features'].to(device)
            )

            labels = batch['label'].to(device)  # One-hot encoded labels

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Compute predictions and accuracy
            normLogits = (logits>0.5).float()
            predictions = normLogits    #torch.argmax(logits, dim=1)  # Class with highest score
            true_labels = labels        #torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            
            correct += (predictions == true_labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

#----------------------------------------------------------------------------------------
def DDP_train(train_data, train_data_swap, valid_data, config,):
    rank = 0
    
    print(f'Process for cuda:{rank} launched.')
    
    predictionModel = Utils.custom_load_model_chkpt(
                        config,
                        checkpointName="Original_notrain",
                        device=f'cuda:{rank}',
                        is_trainable=True
                        )
    
    model = predictionModel
    model.to(f'cuda:{rank}')
    
    optimizer = optim.AdamW([
        {'params': model.gemma_model.parameters(), 'lr': config.base_model_lr},     # Lower learning rate for transformer layers
        {'params': model.feature_fc.parameters(), 'lr': config.feature_fc_lr},      # Higher learning rate for custom layers
        {'params': model.classifier.parameters(), 'lr': config.classifier_lr},      # Higher learning rate for custom layers
    ], weight_decay=0.01)
    #optimizer = optim.Adam(ddp_model.parameters(), weight_decay=0.01)

    num_training_steps = len(train_data) * config.n_epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.27
    )

    min_val_loss = float('inf')
    min_acc = 0
    history = {"train_accum_loss" : [], "train_accum_accuracy" : [], "valid_loss" : [], 
                "valid_accuracy" : []}
    history["best_epoch"]=0
    history["best_loss"]=0
    history["best_acc"]=0

    loss_fn = nn.BCELoss()

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M")
    checkpoint_prefix = date_time+'_'+str(config.max_length)+'_'
    
    print(f'Running {config.n_epochs} epochs on rank {rank}.')
    sys.stdout.flush()
    
    for epoch in range(config.n_epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        if epoch % 2 == 0:
            current_loader = train_data
        else:
            current_loader = train_data_swap
        
        for batch in tqdm(current_loader, total=len(current_loader), unit='row') if rank == 0 else current_loader:
            optimizer.zero_grad()
            
            inputs_ids = batch['input_ids'].to(f'cuda:{rank}')
            attention_mask = batch['attention_mask'].to(f'cuda:{rank}')
            features = batch['features'].to(f'cuda:{rank}')
            
            logits = model(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                features=features
            )
            
            labels = batch['label'].to(f'cuda:{rank}')
        
            loss = loss_fn(logits, labels)
        
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            total_loss += loss.item()
            
            # Compute predictions and accuracy
            normLogits = (logits>0.5).float()
            predictions = normLogits    #torch.argmax(logits, dim=1)  # Class with highest score
            true_labels = labels        #torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            
            correct += (predictions == true_labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(current_loader)
        accuracy = correct / total_samples
        
        if rank==0:
            
            metrics = evaluate_model(model, valid_data, device=f'cuda:{rank}')
            
            # add date and hour + epochs in checkpoint_name
            # Calculate average loss and accuracy
            print(f"Epoch {epoch + 1} Finished")
            print(f"Accumulated Train Loss: {avg_loss}")
            print(f"Accumulated Train Accuracy: {accuracy}")
            print(f"Valid Loss: {metrics['loss']}, Valid Accuracy : {metrics['accuracy']}")

            history['train_accum_loss'].append(avg_loss)
            history['train_accum_accuracy'].append(accuracy)
            history['valid_loss'].append(metrics['loss'])
            history['valid_accuracy'].append(metrics['accuracy'])

            chkptName = checkpoint_prefix + 'train'

            if min_val_loss > metrics['loss']:
                print(f"{metrics['loss']} val loss is better than previous {min_val_loss}, saving checkpoint_lossBest, epoch: ", epoch + 1)
                Utils.custom_save_model_chkpt(model, config, checkpointName=chkptName+"_lossBest", epoch=epoch+1)
                history["best_epoch"] = epoch + 1
                history["best_loss"] = metrics['loss']
                min_val_loss = metrics['loss']

            if min_acc < metrics['accuracy']:
                print(f"{metrics['accuracy']} val accuracy is better than previous {min_acc}, saving checkpoint_accBest, epoch: ", epoch + 1)
                Utils.custom_save_model_chkpt(model, config, checkpointName=chkptName+"_accBest", epoch=epoch+1)
                history["best_epoch"] = epoch + 1
                history["best_acc"] = metrics['accuracy']
                min_acc = metrics['accuracy']

            Utils.save_history(history, config, chkptName+"_lossBest")
            print(f"-----------------------------------------------------------------")
    

#----------------------------------------------------------------------------------------
def script():
    print("Transformers:", trsf.__version__)
    print("Peft:", pft.__version__)
    print('Torch version:', torch.__version__)
    print('Torch is build with CUDA:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_ids = list(range(torch.cuda.device_count()))
    device_ids

    print(f'Torch available devices ids : cuda:{device_ids}')
    print('------------------------------')


    config_file = 'Configs.py'
    manager = Configs.ConfigManager(config_file)

    config = manager.micro

    print(f'config : {config.config_name}')

    base_model_path = config.basemodel_path
    dataframe_path = config.train_data
    dataframe_swapab_path = config.swapab_data

    try:
        df = pd.read_csv(dataframe_path)
    except:
        print(f"Could not load dataframe : {dataframe_path}")
        
    try:
        df_swap = pd.read_csv(dataframe_swapab_path)
    except:
        print(f"Could not load dataframe : {dataframe_path}")

    df = df.sample(frac=config.sample_size, random_state=config.random_seed)
    df_swap = df_swap.sample(frac=config.sample_size, random_state=config.random_seed)

    print(f"Tokenize...")

    df['prompt'] = df['prompt'].astype(str)
    df['response_a'] = df['response_a'].astype(str)
    df['response_b'] = df['response_b'].astype(str)
    df_swap['prompt'] = df_swap['prompt'].astype(str)
    df_swap['response_a'] = df_swap['response_a'].astype(str)
    df_swap['response_b'] = df_swap['response_b'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.add_eos_token = True      # We'll add <eos> at the end
    tokenizer.padding_side = "right"

    def tokenize_df(row):
        return Utils.tokenize(tokenizer, [row['prompt']], [row['response_a']], [row['response_b']], max_length=config.max_length)

    df['tokens'] = df.apply(tokenize_df, axis=1)
    df['len'] = df['prompt_len'] + df['response_a_len'] + df['response_b_len']
    df_swap['tokens'] = df_swap.apply(tokenize_df, axis=1)
    df_swap['len'] = df_swap['prompt_len'] + df_swap['response_a_len'] + df_swap['response_b_len']

    print(f"Train Dataframe of shape : {df.shape}")
    print(f"Tokenize OK")


    print(f"Split and prepare Loaders...")

    df_train, df_valid = train_test_split(df, test_size=config.validation_size, random_state=config.random_seed)
    df_train_swap, df_valid_swap = train_test_split(df_swap, test_size=config.validation_size, random_state=config.random_seed)

    # Prepare dataset and dataloader
    dataset_train = Utils.ChatbotArenaDataset(df_train, tokenizer, max_length=config.max_length)
    dataloader_train = Utils.DataLoader(dataset_train, batch_size=config.train_batch, shuffle=True, num_workers=len(device_ids))

    dataset_valid = Utils.ChatbotArenaDataset(df_valid, tokenizer, max_length=config.max_length)
    dataloader_valid = Utils.DataLoader(dataset_valid, batch_size=config.eval_batch, shuffle=True, num_workers=len(device_ids))
    
    dataset_train_swap = Utils.ChatbotArenaDataset(df_train_swap, tokenizer, max_length=config.max_length)
    dataloader_train_swap = Utils.DataLoader(dataset_train_swap, batch_size=config.train_batch, shuffle=True)

    print(f"Split and prepare Loaders OK")

    print(f"Trainning...")
    
    DDP_train(dataloader_train, dataloader_train_swap, dataloader_valid, config)

    print(f"Trainning OK")

#----------------------------------------------------------------------------------------
if __name__ == '__main__':
    script()