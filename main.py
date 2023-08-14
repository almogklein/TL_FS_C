# ######################## Installations ###############################
import itertools
import os
import json
import random
import time
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from tqdm import tqdm
import joblib
import torch
import pickle
import datetime
import argparse
import data_prep
import util as u
import dataloader
import torch.optim 
import stats as st
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats
import cosin_calc as cc
import infering as infer
import ast_models as ast_mo
from util import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
# #####################################################################


def process_batch(audio_output):
    
    batch_size = audio_output.size(0)
    predictions = audio_output.to('cpu').detach()
    predicted_thresholds = torch.zeros_like(predictions)  # Initialize the tensor with zeros
    
    for i in range(0, batch_size, 5):
        quintile_predictions = predictions[i:i+5]
        
        max_value, _ = torch.max(quintile_predictions, dim=0)
        predicted_thresholds[i:i+5] = (quintile_predictions == max_value).float()
    
    return predicted_thresholds

def sieamis_ast_infer(audio_model, test_loader, cla):
    
    batch_time = AverageMeter()
        
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    
    device = torch.device("cuda:0")
    audio_model = audio_model.to(device)
    
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_predictions_tresholds = []
    
    A_targets = []
    A_real_class = []
    x_d = []
    y_d = []
    
    with torch.no_grad():
        for iii, (audio_input1, audio_input2, labels, real_class) in tqdm(enumerate(test_loader)):
            # if iii < 1000:
            audio_input1 = audio_input1.to(device, non_blocking=True)
            audio_input2 = audio_input2.to(device, non_blocking=True)

            # compute output
            audio_output = audio_model(audio_input1, audio_input2)
            
            predictions = audio_output.to('cpu').detach()
            
            # max_value = predictions.max()  
            # predicted_thresholds = (predictions == max_value).float()
            predicted_thresholds = process_batch(audio_output)
            
            # predicted_thresholds___ = torch.sigmoid(audio_output)  # (predictions > 0.5).float()
            
            x_d.extend([a[0] for a in [np.array(predicted_thresholds).tolist()][0]])
            y_d.extend([a for a in [np.array(labels.to('cpu').detach()).tolist()][0]])
            
        

            # compute the loss
            labels = labels.view(labels.shape[0], -1)
            labels = labels.to(torch.float32)
            labels = labels.to(device, non_blocking=True)
            
            
            A_targets.append(labels)
            A_real_class.append(real_class)
            A_predictions.append(predictions)
            A_predictions_tresholds.append(predicted_thresholds)

            batch_time.update(time.time() - end)
            end = time.time()
            # else:
            #     break
    cla_pred = cla.predict(np.array(x_d).reshape(-1, 1)) 
    
    audio_output = torch.cat(A_predictions_tresholds)
    target = torch.cat(A_targets)
    A_predictions_tresholds = torch.cat(A_predictions_tresholds)
    
    stats = st.calculate_stats_(audio_output, target.to('cpu').detach())

    return stats, target, A_predictions_tresholds, A_predictions, A_real_class, [x_d, y_d, cla_pred]

def preper_data_for_ast_model(parser_param, class_map):
    
    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.set_defaults(**parser_param)
    
    args = parser.parse_args()
    print('now train a audio spectrogram transformer model')

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem,
                  'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean': args.dataset_mean, 'std': args.dataset_std,
                  'noise': args.noise}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0,
                      'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std,
                      'noise': args.noise}
    

    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.ESC_TL_Dataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, class_map=class_map),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.ESC_TL_Dataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, class_map=class_map),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    audio_model = ast_mo.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain)

    print("\nCreating experiment directory: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

    return train_loader, val_loader, audio_model, args

def validate_sieamis(audio_model, loader, device, loss_fun, classifi_modle):
        
        batch_time = AverageMeter()
        
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        
        audio_model = audio_model.to(device, non_blocking=True)
        
        # switch to evaluate mode
        audio_model.eval()

        end = time.time()
        A_predictions = []
        A_predictions_tresholds = []
        x_data_val = []
        y_data_val = []
        
        A_targets = []
        A_loss = []
        A_real_class = []
        rf_pred = []
        
        with torch.no_grad():
            for enu_i, (audio_input1, audio_input2, labels, real_class) in tqdm(enumerate(loader)):
                
                # if enu_i < 100:
                audio_input1 = audio_input1.to(device, non_blocking=True)
                audio_input2 = audio_input2.to(device, non_blocking=True)

                labels = labels.view(labels.shape[0], -1)
                labels = labels.to(torch.float32)
                labels = labels.to(device, non_blocking=True)
                
                # compute output
                audio_output = audio_model(audio_input1, audio_input2)

                # compute the loss
                loss = loss_fun(audio_output.half(), labels.half())
                
                predictions = audio_output.to('cpu').detach()
                # max_value = predictions.max()  
                # predicted_thresholds = (predictions == max_value).float()
                predicted_thresholds = process_batch(audio_output)
                
                x_data_val.extend([a[0] for a in [np.array(predicted_thresholds).tolist()][0]])
                y_data_val.extend([a[0] for a in [np.array(labels.to('cpu').detach()).tolist()][0]])
                
                A_targets.append(labels.to('cpu').detach())
                A_real_class.append(real_class)
                A_predictions.append(predictions)
                A_predictions_tresholds.append(predicted_thresholds)
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()
            # else:
            #     break
            audio_output_ = torch.cat(A_predictions_tresholds)
            audio_output = torch.cat(A_predictions)
            
            target = torch.cat(A_targets)
            loss = np.mean(A_loss)
            
            xgb_classifier = classifi_modle[0]
            rf_classifier = classifi_modle[1]
            gb_classifier = classifi_modle[2]
            svm_classifier = classifi_modle[3]
            
            xgb_y_hat_data_val = xgb_classifier.predict(np.array(x_data_val).reshape(-1, 1)) 
            rf_y_hat_data_val = rf_classifier.predict(np.array(x_data_val).reshape(-1, 1)) 
            gb_y_hat_data_val = gb_classifier.predict(np.array(x_data_val).reshape(-1, 1)) 
            svm_y_hat_data_val = svm_classifier.predict(np.array(x_data_val).reshape(-1, 1)) 
            
            stats = st.calculate_stats_(audio_output_, target)

        return stats, loss, target, audio_output_, A_real_class, audio_output, [x_data_val, y_data_val, [xgb_y_hat_data_val, rf_y_hat_data_val, gb_y_hat_data_val, svm_y_hat_data_val]]
    
def super_vi_ast_cross_val_kfold(epochs, batch_size, cv_folds, class_lookup, k):
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_train_data__{fold_num}_all_{k}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_eval_data__{fold_num}_all_{k}.json", 
            "data_eval": "",
            "label_csv": f"/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices_fold_{k}.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/00/{k}/{fold_num}",
            "lr": 1e-4, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,   
            "imagenet_pretrain": False,
            "audioset_pretrain": False,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
     
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_train_data__{fold_num}_all_{k}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_eval_data__{fold_num}_all_{k}.json", 
            "data_eval": "",
            "label_csv": f"/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices_fold_{k}.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/10/{k}/{fold_num}",
            "lr": 1e-4, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,
            "imagenet_pretrain": True,
            "audioset_pretrain": False,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_train_data__{fold_num}_all_{k}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_cla_datafile/35_esc_eval_data__{fold_num}_all_{k}.json", 
            "data_eval": "",
            "label_csv": f"/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices_fold_{k}.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/11/{k}/{fold_num}",
            "lr": 1e-5, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,
            "imagenet_pretrain": True,
            "audioset_pretrain": True,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
        
def super_vi_ast_cross_val(epochs, batch_size, cv_folds, class_lookup):
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_train_data_{fold_num}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_eval_data_{fold_num}.json", 
            "data_eval": "",
            "label_csv": "/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/00/{fold_num}",
            "lr": 1e-4, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,   
            "imagenet_pretrain": False,
            "audioset_pretrain": False,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
     
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_train_data_{fold_num}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_eval_data_{fold_num}.json", 
            "data_eval": "",
            "label_csv": "/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/10/{fold_num}",
            "lr": 1e-4, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,
            "imagenet_pretrain": True,
            "audioset_pretrain": False,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_train_data_{fold_num}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_eval_data_{fold_num}.json", 
            "data_eval": "",
            "label_csv": "/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_exp/11/{fold_num}",
            "lr": 1e-5, "optim": "adam", "batch_size": batch_size,
            "num_workers": 8, "n_epochs": epochs, "lr_patience": 2,
            "n_print_steps": 100, "save_model": False, "freqm": 24,
            "timem": 96, "mixup": 0, "bal": None, "fstride": 10,
            "tstride": 10,
            "imagenet_pretrain": True,
            "audioset_pretrain": True,
            "dataset_mean": -6.6268077, "dataset_std": 5.358466,
            "audio_length": 512, "noise": False, "metrics": 'acc',
            "loss": 'CE',
            "warmup": True, "lrscheduler_start": 5, "lrscheduler_step": 1,
            "lrscheduler_decay": 0.85, "wa": False, "wa_start": 1, "wa_end": 5
        }
        train_loader, val_loader, audio_model, args = preper_data_for_ast_model(parser_param, class_map=class_lookup)
        train_ast_supervie_lerning(train_loader, val_loader, audio_model, args)
    
def train_ast_supervie_lerning(train_loader, val_loader, audio_model, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, _, best_mAP, best_acc, _ = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    def validate(audio_model, val_loader, args, epoch):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_time = AverageMeter()
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        audio_model = audio_model.to(device)
        # switch to evaluate mode
        audio_model.eval()

        end = time.time()
        A_predictions = []
        A_targets = []
        A_loss = []
        with torch.no_grad():
            for i, (audio_input, labels) in enumerate(val_loader):
                audio_input = audio_input.to(device)

                # compute output
                audio_output, _ = audio_model(audio_input)
                audio_output = torch.sigmoid(audio_output)
                
                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                # compute the loss
                labels = labels.to(device)
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()

            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            loss = np.mean(A_loss)
            stats = st.calculate_stats(audio_output, target)

            # save the prediction here
            exp_dir = args.exp_dir
            if os.path.exists(exp_dir+'/predictions') == False:
                os.mkdir(exp_dir+'/predictions')
                np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
            np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

        return stats, loss

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)    
    args.loss_fn = loss_fn
    
    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    print('scheduler for esc-50 is used')
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
   
    epoch += 1
    scaler = GradScaler()

    print("start training...")
    result = np.zeros([args.n_epochs, 8])
    audio_model.train()

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output, _ = audio_model(audio_input)
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, val_loader, args, epoch)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        
        print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(st.d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

       
        result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, st.d_prime(mAUC), loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))


        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        # with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
        #     pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def preper_data_for_sieamis_ast_model_train_test_val(args, pair_path, batch_size):
    
    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}
    
    
    print('now train a audio spectrogram transformer model')

    train_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem,
                  'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean': args.dataset_mean, 'std': args.dataset_std,
                  'noise': args.noise}
    
    test_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0,
                  'mixup': 0, 'dataset': args.dataset, 'mode':'test', 'mean': args.dataset_mean, 'std': args.dataset_std,
                  'noise': args.noise}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0,
                      'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std,
                      'noise': args.noise}

    print('start loading ESC_FSL_Dataset:\n')
    begin_time_all = time.time()
    train_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[0], audio_conf=train_audio_conf), 
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading train_loader\n')
    print('Loading time: {:.3f}'.format(time.time()-begin_time_all))
    
    begin_time = time.time()
    test_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[1], audio_conf=test_audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading test_loader\n')
    print('Loading time: {:.3f}'.format(time.time()-begin_time))

    begin_time = time.time()
    val_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[2], audio_conf=val_audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading val_loader\n')
    print('Loading time for all: {:.3f}'.format(time.time()-begin_time_all))

    return train_loader, val_loader, test_loader

def run_sieamis_ast(train_loader, val_loader, audio_model, args, models_names, mod_index, cont=False):
    
    print(datetime.datetime.now())
    device = torch.device("cuda:0")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    
    global_step, epoch, best_acc, best_classifier = 0, 0, 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    
    audio_model = audio_model.to(device, non_blocking=True)
    
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format((sum(p.numel() for p in audio_model.parameters()) / 1e6)))
    print('Total trainable parameter number is : {:.3f} million'.format((sum(p.numel() for p in trainables) / 1e6)))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-5, betas=(0.95, 0.999))

    # dataset specific settings
    if cont:
        main_metrics = 'Contrastive Loss'
        loss_fn = ast_mo.ContrastiveLoss(margin=1.0)
        args.loss_fn = loss_fn
    else:
        main_metrics = 'BCE Loss'
        ################################# 
        pos_weight = torch.tensor([4.0])
        pos_weight = pos_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        args.loss_fn = loss_fn
    
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
                                                     gamma=args.lrscheduler_decay, verbose=False)    
    
    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    print('scheduler for esc is used')
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    epoch += 1
    scaler = GradScaler()

    print("\nstart training...")
    result = np.zeros([args.n_epochs, 12])
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        x_data = []
        y_data = []
        
        for i, (audio_input1, audio_input2, labels, _) in tqdm(enumerate(train_loader)):
            # if i < 500:
            B = audio_input1.size(0)
            begin_time_batch = time.time()
            
            audio_input1 = audio_input1.to(device, non_blocking=True)
            audio_input2 = audio_input2.to(device, non_blocking=True)
            
            labels = labels.view(labels.shape[0], -1)
            labels = labels.to(torch.float32)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - begin_time_batch)
            per_sample_data_time.update(((time.time() - begin_time_batch) / audio_input1.shape[0]) * 2)
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 100 and global_step % 100 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))
            
            with autocast():
                optimizer.zero_grad()
                
                if cont:
                    audio_output1, audio_output2 = audio_model(audio_input1, audio_input2)          
                    loss = loss_fn(audio_output1, audio_output2, labels)
                
                else:    
                    audio_output = audio_model(audio_input1, audio_input2)                        
                    loss = loss_fn(audio_output.half(), labels.half())
            
            # predictions = audio_output.to('cpu').detach()
            # max_value = predictions.max()  
            # predicted_thresholds = (predictions == max_value).float()
            predicted_thresholds = process_batch(audio_output)
            
            x_data.extend([a[0] for a in [np.array(predicted_thresholds).tolist()][0]])
            y_data.extend([a[0] for a in [np.array(labels.to('cpu').detach()).tolist()][0]])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - begin_time_batch)
            per_sample_time.update((time.time() - begin_time_batch) / audio_input1.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / audio_input1.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                
                end_time = time.time()
                print('finished batch at:')
                print(f'{end_time - begin_time_batch} sec\t {(end_time - begin_time_batch)/60} min\n')
                
                print('Epoch: [{0}][{1}/{2}]\n'
                'Per Sample Total Time {per_sample_time.avg:.5f}\n'
                'Per Sample Data Time {per_sample_data_time.avg:.5f}\n'
                'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\n'
                'Train Loss {loss_meter.avg:.4f}\n'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            global_step += 1
        # else:
        #     break
        xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=4, scale_pos_weight=4)
        xgb_classifier.fit(np.array(x_data).reshape(-1, 1), y_data)
        
        # Train the Random Forest model
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2, min_samples_leaf=1, class_weight={0:1, 1:4})
        rf_classifier.fit(np.array(x_data).reshape(-1, 1), y_data)
        
        gb_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_split=2, min_samples_leaf=1)
        gb_classifier.fit(np.array(x_data).reshape(-1, 1), y_data)
        
        svm_classifier = SVC(class_weight={0:1, 1:4})
        svm_classifier.fit(np.array(x_data).reshape(-1, 1), y_data)
        
        classifiers = [xgb_classifier, rf_classifier, gb_classifier, svm_classifier]
        
        end_epoch__time1 = time.time()
        print('finished training  epoch at:')
        print(f'{end_epoch__time1 - begin_time} sec\t {(end_epoch__time1 - begin_time)/60} min\n')
        
        print('start validation')
        stats, valid_loss, target, A_predictions_tresholds, A_real_class, A_predictions, classi_data = validate_sieamis(audio_model, val_loader, device, loss_fn, classifiers)
        
        x_, y_, pred_mo  = classi_data
        xgb_pred, rf_pred, gb_pred, svm_pred = pred_mo
        
        balanc_accuracy_b_xgb = balanced_accuracy_score(y_, xgb_pred)
        print('xgb BINARY BALANCED_ACC: ', balanc_accuracy_b_xgb)
        
        balanc_accuracy_b_rf = balanced_accuracy_score(y_, rf_pred)
        print('rf BINARY BALANCED_ACC: ', balanc_accuracy_b_rf)
        
        balanc_accuracy_b_gb = balanced_accuracy_score(y_, gb_pred)
        print('gb BINARY BALANCED_ACC: ', balanc_accuracy_b_gb)
        
        balanc_accuracy_b_svm = balanced_accuracy_score(y_, svm_pred)
        print('svm BINARY BALANCED_ACC: ', balanc_accuracy_b_svm)
        
        
        end_epoch__time = time.time()
        print('finished val  epoch at:')
        print(f'{end_epoch__time - end_epoch__time1} sec\t {(end_epoch__time - end_epoch__time1)/60} min\n')
        
        f1 = stats['f1']
        acc = stats['acc']
        auc = stats['auc']
        AP = stats['AP']
        recall = stats['recalls']
        precision = stats['precisions']
        
        tp, tn, fp, fn = stats['1_t'], stats['0_f'], stats['1_f'], stats['0_f']

        print("acc: {:.6f}".format(acc))
        print("auc: {:.6f}".format(auc))
        print("AP: {:.6f}".format(AP))
        print("precision: {:.6f}".format(precision))
        print("recall: {:.6f}".format(recall))
        print("f1: {:.6f}".format(f1))
    
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, auc, loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr'], precision, recall, f1, tp, tn, fp, fn]
        
        if os.path.exists(exp_dir + f'/{models_names[mod_index]}') == False:
            os.mkdir(exp_dir + f'/{models_names[mod_index]}')
            
        np.savetxt(exp_dir + f'/{models_names[mod_index]}/val_result.csv', result, delimiter=',')
        
        print('validation finished')
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))
        if np.max([balanc_accuracy_b_xgb, balanc_accuracy_b_rf, balanc_accuracy_b_gb, balanc_accuracy_b_svm]) > best_classifier:
            
            best_classifier = np.max([balanc_accuracy_b_xgb, balanc_accuracy_b_rf, balanc_accuracy_b_gb, balanc_accuracy_b_svm])
            best_model_ind = np.argmax([balanc_accuracy_b_xgb, balanc_accuracy_b_rf, balanc_accuracy_b_gb, balanc_accuracy_b_svm])
            best_classi = classifiers[best_model_ind]
            exp_dir_ = exp_dir + f'/{models_names[mod_index]}'
            
            if best_model_ind > 0:
                # Save the Support Vector Machine (SVM) model
                joblib.dump(best_classi, exp_dir_+'/model.pkl')
            else:
                # Save the XGBoost model
                best_classi.save_model(exp_dir_+'/xgboost_model.model')
        
        if acc > best_acc:
            
            exp_dir_ = exp_dir + f'/{models_names[mod_index]}'
            best_acc = acc
            
            torch.save(audio_model.state_dict(), "%s/sieamis_audio_model.pth" % (exp_dir_))
            torch.save(optimizer.state_dict(), "%s/sieamis_optim_state.pth" % (exp_dir_))
            
            with open(exp_dir + f'/{models_names[mod_index]}/data_predict_val.json', 'w') as f:
                json.dump({'target': target.detach().cpu().numpy().tolist(),
                           'A_predictions': [i.detach().cpu().numpy().tolist() for i in A_predictions],
                           'A_real_class': A_real_class}, f, indent=1)
           
        epoch += 1
        scheduler.step()
        data_time.reset()
        batch_time.reset()
        loss_meter.reset()
        per_sample_time.reset()
        per_sample_dnn_time.reset()
        per_sample_data_time.reset()
            
    print(f"\nfinished all {args.n_epochs} training...")
    print('training time: {:.3f}'.format(time.time()-start_time))
    
    return audio_model, optimizer, best_classi

def main():
    
    # Path to ESC-50 master directory (change if needed)
    ESC50_PATH = '/home/almogk/ESC-50-master'
    CODE_REPO_PATH = '/home/almogk/FSL_TL_E_C'

    train_class_names = ['snoring', 'sea_waves', 'rain', 'chainsaw', 'pig', 'brushing_teeth', 'clock_tick', 'mouse_click', 'clapping', 'frog',
                        'door_wood_creaks', 'washing_machine', 'laughing', 'can_opening', 'toilet_flush', 'insects', 'chirping_birds', 'train',
                        'fireworks', 'drinking_sipping', 'water_drops', 'door_wood_knock', 'keyboard_typing', 'wind', 'siren', 'crying_baby', 
                        'car_horn', 'sheep', 'crickets', 'footsteps', 'cat', 'pouring_water', 'helicopter', 'hand_saw', 'rooster']
    test_class_names = ['vacuum_cleaner', 'sneezing', 'thunderstorm', 'crow', 'glass_breaking', 'church_bells', 'cow', 'dog', 'airplane', 'engine']
    val_class_names = ['crackling_fire', 'clock_alarm', 'coughing', 'hen', 'breathing']
    
    class_names = ['snoring', 'sea_waves', 'rain', 'chainsaw', 'pig', 'brushing_teeth', 'clock_tick', 'mouse_click', 'clapping', 'frog',
                    'door_wood_creaks', 'washing_machine', 'laughing', 'can_opening', 'toilet_flush', 'insects', 'chirping_birds', 'train',
                    'fireworks', 'drinking_sipping', 'water_drops', 'door_wood_knock', 'keyboard_typing', 'wind', 'siren', 'crying_baby', 
                    'car_horn', 'sheep', 'crickets', 'footsteps', 'cat', 'pouring_water', 'helicopter', 'hand_saw', 'rooster', 'vacuum_cleaner', 
                    'sneezing', 'thunderstorm', 'crow', 'glass_breaking', 'church_bells', 'cow', 'dog', 'airplane', 'engine', 'crackling_fire', 'clock_alarm', 
                    'coughing', 'hen', 'breathing']
    
    class_lookup_10 = {'00': 0, '03': 1, '09': 2, '19': 3, '21': 4, '36': 5, '39': 6, '44': 7, '46': 8, '47': 9}
    class_lookup_5 = {'06': 0, '12': 1, '23': 2, '24': 3, '37': 4}
    
    class_lookup_35 = {'1': 0, '2': 1, '4': 2, '5': 3, '7': 4, '8': 5, '10': 6, '11': 7, '13': 8, 
                    '14': 9, '15': 10, '16': 11, '17': 12, '18': 13, '20': 14, '22': 15,
                    '25': 16, '26': 17, '27': 18, '28': 19, '29': 20, '30': 21, '31': 22,
                    '32': 23, '33': 24, '34': 25, '35': 26, '38': 27, '40': 28, '41': 29,
                    '42': 30, '43': 31, '45': 32, '48': 33, '49': 34}
    
    train_names_sets, val_names_sets, test_names_sets = u.split_names_with_cross_validation(class_names, CODE_REPO_PATH, 3)
    
    id2label_map = {0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog', 5: 'cat', 6: 'hen', 7: 'insects',
             8: 'sheep', 9: 'crow', 10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets', 14: 'chirping_birds', 15: 'water_drops',
             16: 'wind', 17: 'pouring_water', 18: 'toilet_flush', 19: 'thunderstorm', 20: 'crying_baby', 21: 'sneezing', 22: 'clapping', 23: 'breathing',
             24: 'coughing', 25: 'footsteps', 26: 'laughing', 27: 'brushing_teeth', 28: 'snoring', 29: 'drinking_sipping', 30: 'door_wood_knock', 31: 'mouse_click',
             32: 'keyboard_typing', 33: 'door_wood_creaks', 34: 'can_opening', 35: 'washing_machine', 36: 'vacuum_cleaner', 37: 'clock_alarm', 38: 'clock_tick',
             39: 'glass_breaking', 40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn', 44: 'engine', 45: 'train', 46: 'church_bells', 47: 'airplane',
             48: 'fireworks', 49: 'hand_saw', 50: 'unknown'}
   
    epochs = 15
    batch_size = 20
    cv_folds = 5
    
    prep_kfold_Train_base_model = False
    if prep_kfold_Train_base_model:
        class_lookup_35_for_all_folds = data_prep.get_data_for_ast_model_kfold(ESC50_PATH, CODE_REPO_PATH, train_names_sets, test_names_sets, val_names_sets)
        for k in range(1, len(class_lookup_35_for_all_folds)):
            super_vi_ast_cross_val_kfold(epochs, batch_size, cv_folds, class_lookup_35_for_all_folds[k], k)
        
    
    Train_base_model = False
    if Train_base_model:
        data_prep.get_data_for_ast_model(ESC50_PATH, CODE_REPO_PATH, train_class_names, test_class_names, val_class_names, class_lookup_35)
        super_vi_ast_cross_val(epochs, batch_size, cv_folds, class_lookup_35)

    # set the number of examples per class and number of classes
    n_shot = [1] # [1, 2, 3, 5]
    k_way = [5] # [2, 3, 5]
    
    query_c_num = [1, 5] # number of query classes for each support set
    query_c_size = [1, 4] # number of query examples per query class
    
    # set the sizes of the support sets
    train_support_set_num = [2000, 5000]
    test_support_set_num = [2000, 15000]
    val_support_set_num = [120]
    
    make_task_sets_from_q = False
    if make_task_sets_from_q:
       u.make_task_sets_from_q(k_way, n_shot, train_support_set_num, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num, val_support_set_num)
    
    make_task_sets_from_ss = False
    if make_task_sets_from_ss:
        flag = 'test'
        for k in range(3):
            if flag == 'test':
                support_set_path = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/test/{k}/2000/2000_test_suppotr_sets.json'
                u.make_task_sets_from_ss(support_set_path, k_way, n_shot, CODE_REPO_PATH, k, test_support_set_num[0], flag)
            elif flag == 'val': 
                support_set_path = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/val/{k}/120/120_val_suppotr_sets.json'
                u.make_task_sets_from_ss(support_set_path, k_way, n_shot, CODE_REPO_PATH, k, val_support_set_num, flag)
    
    make_task_sets_from_unknown = False
    if make_task_sets_from_unknown:
        flag = 'val'
        for k in range(3):
            if flag == 'test':
                u.make_task_sets_from_unknown_q(k_way, n_shot, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num, k, text='test', data_path=f'/data_kfold/test_datafile/esc_fsl_test_data_{k}.json')
            elif flag == 'val': 
                u.make_task_sets_from_unknown_q(k_way, n_shot, query_c_num, query_c_size, CODE_REPO_PATH, val_support_set_num,  k, text='val', data_path=f'/data_kfold/val_datafile/esc_fsl_val_data_{k}.json')
    
    EXTRACT_EMBEDDINGS_AST = False
    if EXTRACT_EMBEDDINGS_AST:
        for k in range(3): 
            output_json = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/embeddings_all_output_{k}.json'
            ft_model_dir_pattern = CODE_REPO_PATH + '/ast_class_ex_kfold/{}' + f'/{k}' + '/{}/models/best_audio_model.pth'
            audio_samples_json = CODE_REPO_PATH + f'/data_kfold/data_files/data_{k}.json'
            audio_samples = infer.load_audio_samples(audio_samples_json)   
            
            audio_model_FF = infer.load_ast_tl_no_ft_model(512, False, False)
            audio_model_TF = infer.load_ast_tl_no_ft_model(512, True, False)
            audio_model_TT = infer.load_ast_tl_no_ft_model(512, True, True)
            embeddings_no_ft = infer.extract_embeddings([audio_model_FF, audio_model_TF, audio_model_TT], audio_samples, 'no_FT')
            
            models = infer.load_pt_ft_models(ft_model_dir_pattern, input_tdim=512)
            embeddings = infer.extract_embeddings(models, audio_samples, 'FT')
            
            embeddings_full = u.merge_dictionaries(embeddings_no_ft, embeddings)   
            infer.save_embeddings(embeddings_full, output_json)
            print(f'finished EXTRACT_EMBEDDINGS for fold {k}')
    else:
        embeddings_full_list = []
        for k in range(3): 
            output_json = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/embeddings_all_output_{k}.json' 
            with open(output_json, 'r') as f:
                embeddings_full = json.load(f)
                embeddings_full = {key: np.array(value) for key, value in embeddings_full.items()}
                embeddings_full_list.append(embeddings_full)
        
    models_names = ['scratch', 'scratch T(ESC-35)', 
                    'PT (ImagNet)', 'PT (ImagNet) FT (ESC-35)',
                    'PT (ImagNet, AudioSet)', 'PT (ImagNet, AudioSet) FT (ESC-35)']
    
    INFER = False
    if INFER:
        
        all_pairs = False
        if all_pairs:
            pair_dic = u.make_all_pairs(CODE_REPO_PATH, test_support_set_num, query_c_num, query_c_size, k_way)
        
        save_file_path = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/'
        
        sim_create = False
        if sim_create:
            for k in range(3):
                
                test_pairs_q = u.read_json(save_file_path + f'/test/{k}/2000/10000_test_2000_1C_1PC_task_sets.json')
                test_pairs_no_q = u.read_json(save_file_path + f'/test/{k}/2000/20000_test_2000_ss_task_sets.json')
        
                pairs_q = u.read_json(save_file_path + f'/val/{k}/120/600_val_120_1C_1PC_task_sets.json')
                pairs_no_q = u.read_json(save_file_path + f'/val/{k}/120/1200_val_120_ss_task_sets.json')

                sim_create = False
                if sim_create:
                    
                    cosine_distances = cc.calculate_cosine_distances(embeddings_full_list[k], pairs_q)
                    cosine_distances_no_q = cc.calculate_cosine_distances(embeddings_full_list[k], pairs_no_q)
                    
                    cosine_distances_test = cc.calculate_cosine_distances(embeddings_full_list[k], test_pairs_q)
                    cosine_distances_test_no_q = cc.calculate_cosine_distances(embeddings_full_list[k], test_pairs_no_q)
                    
                    u.write_json(save_file_path + f'/val/{k}/120/600_cos_sim_val_q.json', cosine_distances)
                    u.write_json(save_file_path + f'/val/{k}/120/cos_sim_val_no_q.json', cosine_distances_no_q)
                    
                    u.write_json(save_file_path + f'/test/{k}/2000/cos_sim_test_q.json', cosine_distances_test)
                    u.write_json(save_file_path + f'/test/{k}/15000/cos_sim_tesr_no_q.json', cosine_distances_test_no_q)    
                    
        make_mc_max_plot = False
        if make_mc_max_plot:
            scors_list = []
            for k in range(3):
                cosine_distances_test = u.read_json(save_file_path + f'/test/{k}/2000/cos_sim_test_q.json')
                test_pairs_q = u.read_json(save_file_path + f'/test/{k}/2000/10000_test_2000_1C_1PC_task_sets.json')
                
                mc_p, balanced_accuracy_mc, accuracies_mc, reports_mc, conf_matrices_mc = cc.evaluate_classification_multiclass_closet_max(cosine_distances_test, test_pairs_q)            
                scors_list.append([mc_p, balanced_accuracy_mc, accuracies_mc, reports_mc, conf_matrices_mc])
            
            bala_acc = [ba[1] for ba in scors_list]
            repo_all = [ba[3] for ba in scors_list]
            
            big_array = np.array(bala_acc)
            bala_acc_average_list = np.mean(big_array, axis=0)
            bala_acc_std_list = np.std(big_array, axis=0)
                
            cc.plot_scors(models_names, [bala_acc_average_list.tolist(), bala_acc_std_list.tolist()], 0, save_file_path+f'/test/scors_multic_max_kfold_3fold_2000.png')
        
        gen_treshold = False
        if gen_treshold:
            
            gen_tresh_scors_list = []
            tres_list = []
            for k in range(3):
                cosine_distances_test = u.read_json(save_file_path + f'/test/{k}/15000/cos_sim_test_q.json')
                test_pairs_q = u.read_json(save_file_path + f'/test/{k}/15000/75000_test_15000_1C_1PC_task_sets.json')
                test_pairs_no_q = u.read_json(save_file_path + f'/test/{k}/15000/150000_test_15000_ss_task_sets.json')


                pairs_q = u.read_json(save_file_path + f'/val/{k}/120/3000_val_120_5C_1PC_task_sets.json')
                pairs_no_q = u.read_json(save_file_path + f'/val/{k}/120/1200_val_120_ss_task_sets.json')
                cosine_distances = u.read_json(save_file_path + f'/val/{k}/120/cos_sim_val_q.json')
                
            
                num_thresholds = 3
                balance_acc_list, err_list, acc_list, report_list, f1_list_list, recall_list_list, precision_list_list, balanced_t = [[] for _ in range(8)]
                thresholds = np.linspace(min(cosine_distances), max(cosine_distances), num=num_thresholds)
                
                for _, threshold in enumerate(thresholds):
                    balanc_accuracies_b, accuracies_b, reports_b, conf_matrices_b, eer_values = cc.evaluate_classification_binary_closet(cosine_distances, pairs_q, threshold)            
                
                    precision_list = []
                    recall_list = []
                    f1_list = []
                    
                    for report in reports_b:
                        macro_avg = report['weighted avg']
                        precision = macro_avg['precision']
                        f1_score = macro_avg['f1-score']
                        recall = macro_avg['recall']
                        
                        precision_list.append(precision)
                        recall_list.append(recall)
                        f1_list.append(f1_score)
                    
                    err_list.append(eer_values)
                    f1_list_list.append(f1_list)
                    acc_list.append(accuracies_b)
                    report_list.append(reports_b)
                    recall_list_list.append(recall_list)
                    balance_acc_list.append(balanc_accuracies_b)
                    precision_list_list.append(precision_list)
            
                    
                    PLOT_IND = False
                    if PLOT_IND:
                    
                        cc.calculate_statistics(cosine_distances, pairs_q, embeddings_full, save_file_path+'/stats.csv')
                        cc.plot_scors(models_names, reports_b, save_file_path+f'/__scors_{threshold}.png')
                        
                        for ii in range(len(models_names)):
                            print(f"{models_names[ii]}")
                            print(f"\n binary Accuracy: {accuracies_b[ii]}")
                            print(f"\n binary Confusion Matrix:\n{conf_matrices_b[ii]}")
                            print(f"\n binary Classification Report for:\n{reports_b[ii]}")
                            
                            print(f"\n Multiclass Accuracy: {accuracies_mc[ii]}")
                            print(f"\n Multiclass Confusion Matrix:\n{conf_matrices_mc[ii]}")
                            print(f"\n Multiclass Classification Report for:\n{reports_mc[ii]}")
                            
                            cc.plot_combined(conf_matrices_b[ii], reports_b[ii], models_names[ii], save_file_path+f'/{models_names[ii]}_cosine_binary_plot.png')
                        balanced_t.append(accuracies_b)
                        cc.plot_scors(models_names, reports_b, save_file_path+f'/__scors_{threshold}.png')
                
                best_indices_acc = [np.argmax([inner_list[i] for inner_list in acc_list]) for i in range(len(acc_list[0]))]
                best_indices_eer = [np.argmin([inner_list[i] for inner_list in err_list]) for i in range(len(err_list[0]))]
                
                best_indices_bala_acc = [np.argmax([inner_list[i] for inner_list in balance_acc_list]) for i in range(len(balance_acc_list[0]))]
                best_indices_pre = [np.argmax([inner_list[i] for inner_list in precision_list_list]) for i in range(len(precision_list_list[0]))]
                
                thresholds_best_acc = [thresholds[best_indices_acc[i]][i] for i in range(len(best_indices_acc))]
                thresholds_best_balanc_acc = [thresholds[best_indices_bala_acc[i]][i] for i in range(len(best_indices_bala_acc))]
                thresholds_best_eer = [thresholds[best_indices_eer[i]][i] for i in range(len(best_indices_eer))]
                thresholds_best_pre = [thresholds[best_indices_pre[i]][i] for i in range(len(best_indices_pre))]
                thresholds_list = [thresholds_best_balanc_acc, thresholds_best_acc, thresholds_best_eer, thresholds_best_pre]
                
                bala_acc_best_list = [acc_list[best_indices_bala_acc[i]][i] for i in range(len(best_indices_bala_acc))]
                acc_best_list = [acc_list[best_indices_acc[i]][i] for i in range(len(best_indices_acc))]
                eer_best_list = [err_list[best_indices_eer[i]][i] for i in range(len(best_indices_eer))]
                prec_best_list = [precision_list_list[best_indices_pre[i]][i] for i in range(len(best_indices_pre))]
                
                balanced_accuracies_b_TEST, accuracies_b_TEST, reports_b_TEST, conf_matrices_b_TEST, eer_values_TEST = cc.evaluate_classification_binary_closet(cosine_distances_test, test_pairs_q, thresholds_list[0])
                gen_tresh_scors_list.append(balanced_accuracies_b_TEST)
                tres_list.append(thresholds_list[0])
            
            big_array = np.array(gen_tresh_scors_list)
            bala_acc_average_list = np.mean(big_array, axis=0)
            bala_acc_std_list = np.std(big_array, axis=0)
            
            tres_array = np.array(tres_list)
            tresh_mean = np.mean(tres_array, axis=0)
            
            cc.plot_b_scors(reports_b_TEST, [bala_acc_average_list.tolist(), bala_acc_std_list.tolist()], balanced_accuracies_b_TEST, tresh_mean, models_names, save_file_path+f'/balance_acc_b_closet_test_kfold_fold_3.png', 'balance_acc')
            
        make_ss_tresh = False
        if make_ss_tresh:
            
            score_list_kfold = []
            sig_list_kfold = []
            for k in range(3):
                test_pairs_q = u.read_json(save_file_path + f'/test/{k}/15000/75000_test_15000_1C_1PC_task_sets.json')
                test_pairs_no_q = u.read_json(save_file_path + f'/test/{k}/15000/150000_test_15000_ss_task_sets.json')
                cosine_distances_test = u.read_json(save_file_path + f'/test/{k}/15000/cos_sim_test_q.json')
                cosine_distances_no_q = u.read_json(save_file_path + f'/val/{k}/120/cos_sim_val_no_q.json')

                
                pairs_q = u.read_json(save_file_path + f'/val/{k}/120/3000_val_120_5C_1PC_task_sets.json')
                pairs_no_q = u.read_json(save_file_path + f'/val/{k}/120/1200_val_120_ss_task_sets.json')
                cosine_distances = u.read_json(save_file_path + f'/val/{k}/120/cos_sim_val_q.json')
                cosine_distances_test_no_q = u.read_json(save_file_path + f'/test/{k}/15000/cos_sim_tesr_no_q.json')
                
                save_file1 = save_file_path + f'/val/{k}/120/ss_personal_param_q_val.json'
                save_file2 = save_file_path + f'/val/{k}/120/ss_personal_param_no_q_val.json'
                
                save_file3 = save_file_path + f'/test/{k}/15000/ss_personal_param_q_test.json'
                save_file4 = save_file_path + f'/test/{k}/15000/ss_personal_param_no_q_test.json'
            
                create_ = False
                if create_:
                    perso_ss_param_q_val = u.make_perso_ss_param(cosine_distances, save_file1)                
                    perso_ss_param_no_q_val = u.make_perso_ss_param_no_q(cosine_distances_no_q, save_file2)
                    
                    perso_ss_param_q_test = u.make_perso_ss_param(cosine_distances_test, save_file3)                
                    perso_ss_param_no_q_test = u.make_perso_ss_param_no_q(cosine_distances_test_no_q, save_file4)
                    
                else:
                    perso_ss_param_q_val = u.read_json(save_file1)
                    perso_ss_param_no_q_val = u.read_json(save_file2)
                    
                    perso_ss_param_q_test = u.read_json(save_file3)
                    perso_ss_param_no_q_test = u.read_json(save_file4)
                
                tresh_sig_const = np.linspace(0, 10, num=2000)
                tresh_alfa_const = np.linspace(0, 10, num=2000)
                
                err_list_, acc_list_ = [], []
                acc_list_5_mss, acc_list_5_mad, acc_list_4_mss, acc_list_4_mad, acc_list_10_mss, acc_list_10_mass_ = [[] for _ in range(6)]
                err_list_5_mss, err_list_5_mad, err_list_4_mss, err_list_4_mad, err_list_10_mss, err_list_10_mass_ = [[] for _ in range(6)]
                all_tresh_ = []
                # ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_q]
                binary_ground_truth_val = [pair[0] for pair in pairs_q]
                
                for index_const, (sig, alf) in enumerate(zip(tresh_sig_const, tresh_alfa_const)):
                    ss_tresholds_all, ss_tresholds_all_0, ss_tresholds_no_q_max_sig_std, ss_tresholds_no_q_mean_sig_std, ss_tresholds_all_max_alfa_diff, ss_tresholds_all_0_max_alfa_diff = [[] for _ in range(6)]
                    for i in range(6):
                        
                        ss_tresholds_all.append([mad - sig*std for mad, std in zip(perso_ss_param_q_val['max'][i], perso_ss_param_q_val['std_all'][i])])                    
                        ss_tresholds_all_max_alfa_diff.append([p95 + alf*diff_ for p95, diff_ in zip(perso_ss_param_q_val['MAD'][i], perso_ss_param_q_val['f_s_dif'][i])])

                        ss_tresholds_all_0.append([mean + sig*std for mean, std in zip(perso_ss_param_q_val['max0'][i], perso_ss_param_q_val['std_0'][i])])
                        ss_tresholds_all_0_max_alfa_diff.append([max_ + alf*diff_ for max_, diff_ in zip(perso_ss_param_q_val['MAD_0'][i], perso_ss_param_q_val['f_s_dif'][i])])
                        
                        ss_tresholds_no_q_mean_sig_std.append([value for value in [mean[0] + sig*std[0] for mean, std in zip(perso_ss_param_no_q_val['max'][i], perso_ss_param_no_q_val['std_all'][i])] for _ in range(query_c_num[1])])
                        ss_tresholds_no_q_max_sig_std.append([max_[0] + sig*std for max_, std in zip([value for val in perso_ss_param_no_q_val['MAD'][i] for value in [val] * query_c_num[1]], perso_ss_param_q_val['f_s_dif'][i])])

                    bc_p_mss, accuracies_b_mss, reports_b_mss, conf_matrices_b_mss, eer_mss, bala_acc_mss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)
                    bc_p_MAD, accuracies_b_mad, reports_b_mad, _, eer_mad, bala_acc_mad = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_max_alfa_diff, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)

                    bc_p_0_mss, accuracies_b_0_mss, reports_b_0_mss, _, eer_0_mss, bala_acc_0_mss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_0, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)
                    bc_p_0_MAD, accuracies_b_0_mad, reports_b_0_mad, _, eer_0_mad, bala_acc_0_mad = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_0_max_alfa_diff, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)
                    
                    bc_p_ss, accuracies_b_per_ss, reports_b_ss, _, eer_ss, bala_acc_ss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_no_q_mean_sig_std, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)
                    bc_p_0_ss, accuracies_b_per_ss_max, reports_b_ss_max_, _, eer_ss_, bala_acc_ss_max = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_no_q_max_sig_std, val_support_set_num[0], k_way[0], True, query_c_num[1], query_c_size[0], binary_ground_truth_val)
                    
                    err_list_.append([eer_mss, eer_0_mss, eer_mad, eer_0_mad, eer_ss, eer_ss_])
                    acc_list_.append([accuracies_b_mss, accuracies_b_mad, accuracies_b_0_mss, accuracies_b_0_mad, accuracies_b_per_ss, accuracies_b_per_ss_max])
                    
                    acc_list_5_mss.append(bala_acc_mss)
                    acc_list_5_mad.append(bala_acc_mad)
                    acc_list_4_mss.append(bala_acc_0_mss)
                    acc_list_4_mad.append(bala_acc_0_mad)
                    acc_list_10_mss.append(bala_acc_ss)
                    acc_list_10_mass_.append(bala_acc_ss_max)
                    
                    err_list_5_mss.append(eer_mss)
                    err_list_5_mad.append(eer_mad)
                    err_list_4_mss.append(eer_0_mss)
                    err_list_4_mad.append(eer_0_mad)
                    err_list_10_mss.append(eer_ss)
                    err_list_10_mass_.append(eer_ss_)
                    
                    all_tresh_.append([sig, alf])
                    
                
                best_indices_acc_mss = [[np.argmax([inner_list[i] for inner_list in acc_list_5_mss]), np.max([inner_list[i] for inner_list in acc_list_5_mss])] for i in range(len(acc_list_5_mss[0]))]
                best_indices_acc_mad = [[np.argmax([inner_list[i] for inner_list in acc_list_5_mad]), np.max([inner_list[i] for inner_list in acc_list_5_mad])] for i in range(len(acc_list_5_mad[0]))]
                
                best_indices_acc_0_mss = [[np.argmax([inner_list[i] for inner_list in acc_list_4_mss]), np.max([inner_list[i] for inner_list in acc_list_4_mss])] for i in range(len(acc_list_4_mss[0]))]
                best_indices_acc_0_mad = [[np.argmax([inner_list[i] for inner_list in acc_list_4_mad]), np.max([inner_list[i] for inner_list in acc_list_4_mad])] for i in range(len(acc_list_4_mad[0]))]
                
                best_indices_acc_per_ss = [[np.argmax([inner_list[i] for inner_list in acc_list_10_mss]), np.max([inner_list[i] for inner_list in acc_list_10_mss])] for i in range(len(acc_list_10_mss[0]))]
                best_indices_acc_ss_max = [[np.argmax([inner_list[i] for inner_list in acc_list_10_mass_]), np.max([inner_list[i] for inner_list in acc_list_10_mass_])] for i in range(len(acc_list_10_mass_[0]))]

                best_indices_err_mss = [[np.argmin([inner_list[i] for inner_list in err_list_5_mss]), np.min([inner_list[i] for inner_list in err_list_5_mss])] for i in range(len(err_list_5_mss[0]))]
                best_indices_err_mad = [[np.argmin([inner_list[i] for inner_list in err_list_5_mad]), np.min([inner_list[i] for inner_list in err_list_5_mad])] for i in range(len(err_list_5_mad[0]))]
                
                best_indices_err_0_mss = [[np.argmin([inner_list[i] for inner_list in err_list_4_mss]), np.min([inner_list[i] for inner_list in err_list_4_mss])] for i in range(len(err_list_4_mss[0]))]
                best_indices_err_0_mad = [[np.argmin([inner_list[i] for inner_list in err_list_4_mad]), np.min([inner_list[i] for inner_list in err_list_4_mad])] for i in range(len(err_list_4_mad[0]))]
                
                best_indices_err_per_ss = [[np.argmin([inner_list[i] for inner_list in err_list_10_mss]), np.min([inner_list[i] for inner_list in err_list_10_mss])] for i in range(len(err_list_10_mss[0]))]
                best_indices_err_ss_max = [[np.argmin([inner_list[i] for inner_list in err_list_10_mass_]), np.min([inner_list[i] for inner_list in err_list_10_mass_])] for i in range(len(err_list_10_mass_[0]))]


                sig_5_eer = [all_tresh_[best_indices_err_mss[i][0]][0] for i in range(len(best_indices_err_mss))]
                alfa_5_eer = [all_tresh_[best_indices_err_mad[i][0]][1] for i in range(len(best_indices_err_mad))]
                
                sig_4_eer = [all_tresh_[best_indices_err_0_mss[i][0]][0] for i in range(len(best_indices_err_0_mss))]
                alfa_4_eer = [all_tresh_[best_indices_err_0_mad[i][0]][1] for i in range(len(best_indices_err_0_mad))]
                
                sig_10_eer = [all_tresh_[best_indices_err_per_ss[i][0]][0] for i in range(len(best_indices_err_per_ss))]
                sig_10__eer= [all_tresh_[best_indices_acc_ss_max[i][0]][0] for i in range(len(best_indices_err_ss_max))]
                
                sig_alfa_eer = [sig_5_eer, alfa_5_eer, sig_4_eer, alfa_4_eer, sig_10_eer, sig_10__eer]
                
                sig_5 = [all_tresh_[best_indices_acc_mss[i][0]][0] for i in range(len(best_indices_acc_mss))]
                alfa_5 = [all_tresh_[best_indices_acc_mad[i][0]][1] for i in range(len(best_indices_acc_mad))]
                
                sig_4 = [all_tresh_[best_indices_acc_0_mss[i][0]][0] for i in range(len(best_indices_acc_0_mss))]
                alfa_4 = [all_tresh_[best_indices_acc_0_mad[i][0]][1] for i in range(len(best_indices_acc_0_mad))]
                
                sig_10 = [all_tresh_[best_indices_acc_per_ss[i][0]][0] for i in range(len(best_indices_acc_per_ss))]
                sig_10_ = [all_tresh_[best_indices_acc_ss_max[i][0]][0] for i in range(len(best_indices_acc_ss_max))]
                
                sig_alfa_acc = [sig_5, alfa_5, sig_4, alfa_4, sig_10, sig_10_]
                sig_alfa_list = [sig_alfa_acc]
                
                # ss_true_labels_test = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_q]
                binary_ground_truth_test = [pair[0] for pair in test_pairs_q]
                for  _, sig_alfa in enumerate(sig_alfa_list):
                    
                    ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_max_sig_std_test, ss_tresholds_no_q_mean_sig_std_test, ss_tresholds_all_max_alfa_diff_test, ss_tresholds_all_0_max_alfa_diff_test = [[] for _ in range(6)]
                    for i in range(6):
                        
                        ss_tresholds_all_test.append([mad - sig_alfa[0][i]*std for mad, std in zip(perso_ss_param_q_test['max'][i], perso_ss_param_q_test['std_all'][i])])                    
                        ss_tresholds_all_max_alfa_diff_test.append([p95 + sig_alfa[1][i]*diff_ for p95, diff_ in zip(perso_ss_param_q_test['MAD'][i], perso_ss_param_q_test['f_s_dif'][i])])
                        
                        ss_tresholds_all_0_test.append([mean + sig_alfa[2][i]*std for mean, std in zip(perso_ss_param_q_test['max0'][i], perso_ss_param_q_test['std_0'][i])])
                        ss_tresholds_all_0_max_alfa_diff_test.append([max_ + sig_alfa[3][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['MAD_0'][i], perso_ss_param_q_test['f_s_dif'][i])])
                        
                        ss_tresholds_no_q_mean_sig_std_test.append([mean[0] + sig_alfa[4][i]*std[0] for mean, std in zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_no_q_test['std_all'][i])])
                        ss_tresholds_no_q_max_sig_std_test.append([max_[0] + sig_alfa[5][i]*std for max_, std in zip(perso_ss_param_no_q_test['MAD'][i], perso_ss_param_q_test['f_s_dif'][i])])

                    bc_p_mss, accuracies_b_test5_mss, reports_b_mss, conf_matrices_b_mss, eer_test5_mss, BA_test5_mss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    bc_p_MAD, accuracies_b_test5_mad, reports_b_mad, conf_matrices_b_mss_, eer_test5_mad, BA_test5_mad = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_max_alfa_diff_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    
                    bc_p_0_mss, accuracies_b_test4_0_mss, reports_b_0_mss, conf_matrices_b_mss__, eer_test4_0_mss, BA_test4_0_mss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_0_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    bc_p_0_MAD, accuracies_b_test4_0_mad, reports_b_0_mad, conf_matrices_b_mss____, eer_test4_0_mad, BA_test4_0_mad = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_0_max_alfa_diff_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    
                    bc_p_ss, accuracies_b_test10, reports_b_ss, conf_matrices_b_mss__________________, eer_test10_ss, BA_test10_ss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_no_q_mean_sig_std_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    bc_p_0_ss, accuracies_b_test10_max, reports_b_ss_max_, conf_matrices_b_mss_____________________________, eer_test10_ss_, BA_test10_ss_ = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_no_q_max_sig_std_test, test_support_set_num[-1], k_way[0], True, query_c_num[0], query_c_size[0], binary_ground_truth_test)
                    
                    score_list_kfold.append([BA_test5_mss, BA_test5_mad, BA_test4_0_mss, BA_test4_0_mad, BA_test10_ss, BA_test10_ss_])
                    sig_list_kfold.append(sig_alfa)
            
            big_array = np.array(score_list_kfold)
            big_array_5, big_array_4, big_array_10 = big_array 
            big_array_5, big_array_4, big_array_10 = big_array_5[::2], big_array_4[::2], big_array_10[::2]
            
            bala_acc_average_list_5 = np.mean(big_array_5, axis=0)
            bala_acc_std_list_5 = np.std(big_array_5, axis=0)
            
            bala_acc_average_list_4 = np.mean(big_array_4, axis=0)
            bala_acc_std_list_4 = np.std(big_array_4, axis=0)
            
            bala_acc_average_list_10 = np.mean(big_array_10, axis=0)
            bala_acc_std_list_10 = np.std(big_array_10, axis=0)
            
            tresh = []
            for sublist in sig_list_kfold:
                new_sublist = []
                for inner_list in sublist[::2]:
                    new_sublist.append(inner_list)
                tresh.append(new_sublist)
            
            tresh_5 =  np.mean([item[0] for item in tresh], axis=0)
            tresh_4 = np.mean([item[1] for item in tresh], axis=0)
            tresh_10 = np.mean([item[2] for item in tresh], axis=0)
            
            tresh_mean = [tresh_5[1:], tresh_4[1:], tresh_10[1:]]
            cc.plot_ss_scors([bala_acc_average_list_5[1:], bala_acc_std_list_5[1:]], 
                                [bala_acc_average_list_4[1:], bala_acc_std_list_4[1:]], 
                                [bala_acc_average_list_10[1:], bala_acc_std_list_10[1:]], tresh_mean, models_names[1:], save_file_path+f'/test/scors_tresholds_per_ss_test_kfold_3fold.png')

    INFER_OPENSET = False
    if INFER_OPENSET:
        
        save_file_path = CODE_REPO_PATH + f'/data_kfold/FSL_SETS/5w_1s_shot/'
        
        scor_kfold_list = []
        tresh_kfold_list = []

        scor_kfold_list_5_A = []
        scor_kfold_list_5_B = []
    
        scor_kfold_list_4_A = []
        scor_kfold_list_4_B = []
        
        scor_kfold_list_10_A = []
        scor_kfold_list_10_B = []
        
        for k in range(3):
                
            test_pairs_openset = u.read_json(save_file_path + f'/test/{k}/15000/75000_test_15000_1C_1PC_task_sets_openset.json')
            test_pairs_no_q = u.read_json(save_file_path + f'/test/{k}/15000/150000_test_15000_ss_task_sets.json')
    
            pairs_openset = u.read_json(save_file_path + f'/val/{k}/120/3000_val_120_5C_1PC_task_sets_openset.json')
            pairs_no_q = u.read_json(save_file_path + f'/val/{k}/120/1200_val_120_ss_task_sets.json')

            take_or_create = True
            if take_or_create:
                
                cosine_distances_openset = u.read_json(save_file_path + f'/val/{k}/120/cosin_openset_val.json')
                cosine_distances_no_q = u.read_json(save_file_path + f'/val/{k}/120/cos_sim_val_no_q.json')
                
                test_cosine_distances_openset = u.read_json(save_file_path + f'/test/{k}/15000/cosin_openset_test.json')        
                cosine_distances_test_no_q = u.read_json(save_file_path + f'/test/{k}/15000/cos_sim_tesr_no_q.json')
            else:    
                
                cosine_distances_openset = cc.calculate_cosine_distances(embeddings_full_list[k], pairs_openset)                
                u.write_json(save_file_path + f'/val/{k}/120/cosin_openset_val.json', cosine_distances_openset)
                
                test_cosine_distances_openset = cc.calculate_cosine_distances(embeddings_full, test_pairs_openset)
                u.write_json(save_file_path + f'/test/{k}/15000/cosin_openset_test.json', test_cosine_distances_openset)
                
                # cosine_distances_no_q = cc.calculate_cosine_distances(embeddings_full, pairs_no_q)
                # u.write_json(save_file_path + f'/val/{k}/120/cos_sim_val_no_q.json', cosine_distances_no_q)
                
                # cosine_distances_test_no_q = cc.calculate_cosine_distances(embeddings_full, test_pairs_no_q)
                # u.write_json(save_file_path + f'/test/{k}/15000/cos_sim_tesr_no_q.json', cosine_distances_test_no_q)
            
            save_file1 = save_file_path + f'/val/{k}/120/ss_personal_param_q_openset_val.json'
            save_file2 = save_file_path + f'/val/{k}/120/ss_personal_param_no_q_val.json'
            
            save_file3 = save_file_path + f'/test/{k}/15000/ss_personal_param_q_openset_test.json'
            save_file4 = save_file_path + f'/test/{k}/15000/ss_personal_param_no_q_test.json'
            create_ = False
            if create_:
                perso_ss_param_q_val = u.make_perso_ss_param(cosine_distances_openset, save_file1)                
                # perso_ss_param_no_q_val = u.make_perso_ss_param_no_q(cosine_distances_no_q, save_file2)
                
                perso_ss_param_q_test = u.make_perso_ss_param(test_cosine_distances_openset, save_file3)                
                # perso_ss_param_no_q_test = u.make_perso_ss_param_no_q(cosine_distances_test_no_q, save_file4) 
            else:            
                perso_ss_param_q_val = u.read_json(save_file1)        
                perso_ss_param_no_q_val = u.read_json(save_file2)            
            
                perso_ss_param_q_test = u.read_json(save_file3)
                perso_ss_param_no_q_test = u.read_json(save_file4)            
            
            
            open_set_fix_tresh = False
            if open_set_fix_tresh:
                
                thresholds = cc.calculate_fix_threshold_openset(cosine_distances_openset, [pair[0] for pair in pairs_openset], 1000)

                # Perform multiclass & binary  classification
                ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
                multiclass_predictions, binary_predictions = cc.multiclass_binary_openset_classification(test_cosine_distances_openset, thresholds, ss_true_labels)
                multiclass_predictions_from_start = []
                for i in range(len(multiclass_predictions)):
                    multiclass_predictions_from_start.append([id2label_map[int(cla[0])] for cla in multiclass_predictions[i][-1]])
                
                binary_ground_truth = [pair[0] for pair in test_pairs_openset]
                multi_ground_truth = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
                multi_ground_truth_from_start = [cla if cla == 'unknown' else id2label_map[int(cla)] for cla in multi_ground_truth]
                
                accuracies_b, reports_b, conf_matrices_b, binary_bala_acc, accuracies_m, reports_m, conf_matrices_m, mc_bala_acc, class_labels = cc.evaluate_classification_openset(multiclass_predictions_from_start, binary_predictions, multi_ground_truth_from_start, binary_ground_truth)
                tresh_fin = [tr[0] for tr in thresholds]
                scor_kfold_list.append(mc_bala_acc)
                tresh_kfold_list.append(tresh_fin)
                
                # cc.plot_confusion_matrices(conf_matrices_m, save_file_path+'/test', class_labels, k, single_plot=True)

                scor_kfold_array = np.array(scor_kfold_list)
                bala_acc_average_list = np.mean(scor_kfold_array, axis=0)
                bala_acc_std_list = np.std(scor_kfold_array, axis=0)
                
                tresh_kfold_array = np.array(tresh_kfold_list)
                tresh_kfold_average_list = np.mean(tresh_kfold_array, axis=0)
                
                cc.plot_scors(models_names, [bala_acc_average_list.tolist(), bala_acc_std_list.tolist()], tresh_kfold_average_list, save_file_path+f'/test/bala_acc_scors_mc_OPENSET_fix_tresh.png')
        
            open_set_personal_tresh = False
            if open_set_personal_tresh:
                
                ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_openset]
                binary_ground_truth_val = [pair[0] for pair in pairs_openset]
                multi_ground_truth_num_val = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in pairs_openset][::5]
                multi_ground_truth_cat_val = [cla if cla == 'unknown' else id2label_map[int(cla)] for cla in multi_ground_truth_num_val]
                class_labels_val = list(set(multi_ground_truth_cat_val))
                class_labels_val.remove('unknown')
                class_labels_val.append('unknown')
                
                thresholds_param = cc.calculate_pers_threshold_openset(cosine_distances_openset, cosine_distances_no_q, [pairs_openset, pairs_no_q], [perso_ss_param_no_q_val, perso_ss_param_q_val], 
                                                                       models_names, val_support_set_num, k_way, query_c_num, query_c_size, ss_true_labels_val, id2label_map, binary_ground_truth_val, 
                                                                       multi_ground_truth_cat_val, class_labels_val, num_thresholds=100)
                
                binary_ground_truth = [pair[0] for pair in test_pairs_openset]  
                multi_ground_truth_num = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
                multi_ground_truth_cat = [cla if cla == 'unknown' else id2label_map[int(cla)] for cla in multi_ground_truth_num]
                class_labels = list(set(multi_ground_truth_cat))
                class_labels.remove('unknown')
                class_labels.append('unknown')
                ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
                
                for  ac_i, sig_alfa in enumerate(thresholds_param):
                    ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_max_sig_std_test, ss_tresholds_no_q_mean_sig_std_test, ss_tresholds_all_max_alfa_diff_test, ss_tresholds_all_0_max_alfa_diff_test = [[] for _ in range(6)]
                    for i in range(len(models_names)):
                        
                        ss_tresholds_all_test.append([mean + sig_alfa[0][i]*std for mean, std in zip(perso_ss_param_q_test['MAD'][i], perso_ss_param_q_test['std_all'][i])])                    
                        # ss_tresholds_all_max_alfa_diff_test.append([max_ - sig_alfa[1][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])
                        
                        ss_tresholds_all_0_test.append([mean + sig_alfa[1][i]*std for mean, std in zip(perso_ss_param_q_test['MAD_0'][i], perso_ss_param_q_test['std_0'][i])])
                        # ss_tresholds_all_0_max_alfa_diff_test.append([max_ + sig_alfa[3][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['max0'][i], perso_ss_param_q_test['f_s_dif'][i])])
                        
                        ss_tresholds_no_q_mean_sig_std_test.append([mean[0] + sig_alfa[2][i]*std[0] for mean, std in zip(perso_ss_param_no_q_test['MAD'][i], perso_ss_param_no_q_test['std_all'][i])])
                        # ss_tresholds_no_q_max_sig_std_test.append([max_[0] + sig_alfa[5][i]*std for max_, std in zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])
                    
                    multiclass_prediction_5_mss, binary_predictions_5_mss, incd_all_5_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    # multiclass_prediction_5_mad, binary_predictions_5_mad, incd_all_5_mad = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_max_alfa_diff_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    
                    multiclass_prediction_4_mss, binary_predictions_4_mss, incd_all_4_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    # multiclass_prediction_4_mad, binary_predictions_4_mad, incd_all_4_mad = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_max_alfa_diff_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    
                    multiclass_prediction_10_mss, binary_predictions_10_mss, incd_all_10_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_mean_sig_std_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    # multiclass_prediction_10_maxss, binary_predictions_10_maxss, incd_all_10_maxss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_max_sig_std_test, test_support_set_num[-1], k_way[0], query_c_num[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                    
                    # cc.plot_scors(models_names, incd_all_10_maxss, incd_all_10_maxss, save_file_path+f'/{ac_eer_i}_scors_mc_OPENSET______.png')
                    # cc.plot_confusion_matrices(conf_matrices_m, save_file_path, class_labels, single_plot=False)
                    
                    scor_kfold_list_5_A.append([elem[-1] for elem in incd_all_5_mss])
                    scor_kfold_list_4_A.append([elem[-1] for elem in incd_all_4_mss])
                    scor_kfold_list_10_A.append([elem[-1] for elem in incd_all_10_mss])                    
                    
                    tresh_kfold_list.append(sig_alfa)
        
                    tresh_kfold_array = np.array(tresh_kfold_list)
                    tresh_kfold_average_list = np.mean(tresh_kfold_array, axis=0)
                                
                    scor_kfold_array_5_A = np.array(scor_kfold_list_5_A)
                    bala_acc_average_list_5_A, bala_acc_std_list_5_A = np.mean(scor_kfold_array_5_A, axis=0).tolist(), np.std(scor_kfold_array_5_A, axis=0).tolist() 
                    
                    scor_kfold_array_4_A = np.array(scor_kfold_list_4_A)
                    bala_acc_average_list_4_A, bala_acc_std_list_4_A = np.mean(scor_kfold_array_4_A, axis=0).tolist(), np.std(scor_kfold_array_4_A, axis=0).tolist() 
                    
                    scor_kfold_array_10_A = np.array(scor_kfold_list_10_A)
                    bala_acc_average_list_10_A, bala_acc_std_list_10_A = np.mean(scor_kfold_array_10_A, axis=0).tolist(), np.std(scor_kfold_array_10_A, axis=0).tolist()
                    
                    cc.plot_ss_scors([bala_acc_average_list_5_A[1:], bala_acc_std_list_5_A[1:]], 
                                    [bala_acc_average_list_4_A[1:], bala_acc_std_list_4_A[1:]], 
                                    [bala_acc_average_list_10_A[1:], bala_acc_std_list_10_A[1:]], 
                                    tresh_kfold_average_list, models_names[1:], save_file_path+f'/test/scors_mc_personal_tresh_openset_test____.png')
            
            
            open_set_personal_CAT_tresh = False
            if open_set_personal_CAT_tresh:
                
                ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_openset]
                binary_ground_truth_val = [pair[0] for pair in pairs_openset]
                multi_ground_truth_num_val = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in pairs_openset][::5]
                multi_ground_truth_cat_val = [cla if cla == 'unknown' else id2label_map[int(cla)] for cla in multi_ground_truth_num_val]
                class_labels_val = list(set(multi_ground_truth_cat_val))
                class_labels_val.remove('unknown')
                class_labels_val.append('unknown')
                
                sig_max, sig_mean, sig_median, ind_max = cc.calculate_pers_CAT_threshold_openset(cosine_distances_openset, cosine_distances_no_q, [pairs_openset, pairs_no_q], 
                                                                                                 [perso_ss_param_no_q_val, perso_ss_param_q_val], models_names, val_support_set_num, 
                                                                                                 k_way, query_c_num, query_c_size, ss_true_labels_val, id2label_map, binary_ground_truth_val, 
                                                                                                 multi_ground_truth_cat_val, class_labels_val, 150)
                        
                binary_ground_truth = [pair[0] for pair in test_pairs_openset]  
                multi_ground_truth_num = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
                multi_ground_truth_cat = [cla if cla == 'unknown' else id2label_map[int(cla)] for cla in multi_ground_truth_num]
                class_labels = list(set(multi_ground_truth_cat))
                class_labels.remove('unknown')
                class_labels.append('unknown')
                ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
                
                ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_mean_sig_std_test = [[] for _ in range(3)]
                for i in range(len(models_names)):
                    for cat_ind in range(1, k_way[0]+1):
                        ss_tresholds_all_test.append([mean[cat_ind] + sig_mean[i]*std[cat_ind] for  _, (mean, std) in enumerate(zip(perso_ss_param_no_q_test['mean_all'][i], perso_ss_param_no_q_test['std_all'][i]))])                    
                        ss_tresholds_all_0_test.append([Max[cat_ind] + sig_max[i]*std[cat_ind] for  _, (Max, std) in enumerate(zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_no_q_test['std_all'][i]))])
                        ss_tresholds_no_q_mean_sig_std_test.append([mad[cat_ind] + sig_median[i]*std[cat_ind] for  _, (mad, std) in enumerate(zip(perso_ss_param_no_q_test['MAD'][i], perso_ss_param_no_q_test['std_all'][i]))])
                
                multiclass_prediction_5_mss, binary_predictions_5_mss, incd_all_5_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_test, test_support_set_num[-1], k_way[0], query_c_num[0], query_c_size[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                multiclass_prediction_4_mss, binary_predictions_4_mss, incd_all_4_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_test, test_support_set_num[-1], k_way[0], query_c_num[0], query_c_size[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                multiclass_prediction_10_mss, binary_predictions_10_mss, incd_all_10_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_mean_sig_std_test, test_support_set_num[-1], k_way[0], query_c_num[0], query_c_size[0], ss_true_labels, id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
        
                # cc.plot_ss_scors([elem[-1] for elem in incd_all_5_mss], [elem[-1] for elem in incd_all_4_mss], [elem[-1] for elem in incd_all_10_mss], [sig_max, sig_mean, sig_median], models_names, save_file_path+f'/PER_CAT_______scors_mc_tresholds_openset_test.png')
                # cc.plot_scors(models_names, incd_all_10_maxss, incd_all_10_maxss, save_file_path+f'/{ac_eer_i}_scors_mc_OPENSET______.png')
                # cc.plot_confusion_matrices(conf_matrices_m, save_file_path, class_labels, single_plot=False)
                
                scor_kfold_list_5_A.append([elem[-1] for elem in incd_all_5_mss])
                scor_kfold_list_4_A.append([elem[-1] for elem in incd_all_4_mss])
                scor_kfold_list_10_A.append([elem[-1] for elem in incd_all_10_mss])                    
                
                tresh_kfold_list.append([sig_mean, sig_max,  sig_median])
                print(f'\nfinished fold {k}................\n')
                            
                tresh_kfold_array = np.array(tresh_kfold_list)
                tresh_kfold_average_list = np.mean(tresh_kfold_array, axis=0)
                            
                scor_kfold_array_5_A = np.array(scor_kfold_list_5_A)
                bala_acc_average_list_5_A, bala_acc_std_list_5_A = np.mean(scor_kfold_array_5_A, axis=0).tolist(), np.std(scor_kfold_array_5_A, axis=0).tolist() 
                
                scor_kfold_array_4_A = np.array(scor_kfold_list_4_A)
                bala_acc_average_list_4_A, bala_acc_std_list_4_A = np.mean(scor_kfold_array_4_A, axis=0).tolist(), np.std(scor_kfold_array_4_A, axis=0).tolist() 
                
                scor_kfold_array_10_A = np.array(scor_kfold_list_10_A)
                bala_acc_average_list_10_A, bala_acc_std_list_10_A = np.mean(scor_kfold_array_10_A, axis=0).tolist(), np.std(scor_kfold_array_10_A, axis=0).tolist()
                
                cc.plot_ss_scors([bala_acc_average_list_5_A[1:], bala_acc_std_list_5_A[1:]], 
                                [bala_acc_average_list_4_A[1:], bala_acc_std_list_4_A[1:]], 
                                [bala_acc_average_list_10_A[1:], bala_acc_std_list_10_A[1:]], 
                                tresh_kfold_average_list, models_names[1:], save_file_path+f'/test/scors_mc_personal_CAT_tresh_openset_test.png')

    epochs = 15
    batch_size = 15
    SIEAMISE = False
    if SIEAMISE:
        
        for k in range(3):
        
            exp_dir = f"/home/almogk/FSL_TL_E_C/sieamis_ast_exp_kfold_2000/{k}/"
            
            parser_param = {
                "n_class": [35], "model": ["ast"],
                'fc': ['m'], 'imagenet_pretrain': [True],'audioset_pretrain': [False, True],
                "dataset": ["esc50"], "exp_dir": [exp_dir],
                "lr": [1e-5], "optim": ["adam"], "batch_size": [batch_size],
                "num_workers": [8], "n_epochs": [epochs], "lr_patience": [2],
                "n_print_steps": [1000], "save_model": [False], "freqm": [24],
                "timem": [96], "mixup": [0], "bal": [None], "fstride": [10],
                "tstride": [10],'fin': [False],
                "dataset_mean": [-6.6268077], "dataset_std": [5.358466],
                "audio_length": [512], "noise": [False], "metrics": ['acc'],
                "loss": ['BCE'],
                "warmup": [True], "lrscheduler_start": [1], "lrscheduler_step": [1],
                "lrscheduler_decay": [0.85], "wa": [False], "wa_start": [1], "wa_end": [5]
                }
            
            parser_param1 = {
                "n_class": [35], "model": ["ast"],
                'fc': ['m'], 'imagenet_pretrain': [False],'audioset_pretrain': [False],
                "dataset": ["esc50"], "exp_dir": [exp_dir],
                "lr": [1e-5], "optim": ["adam"], "batch_size": [batch_size],
                "num_workers": [8], "n_epochs": [epochs], "lr_patience": [2],
                "n_print_steps": [1000], "save_model": [False], "freqm": [24],
                "timem": [96], "mixup": [0], "bal": [None], "fstride": [10],
                "tstride": [10], 'fin': [True],
                "dataset_mean": [-6.6268077], "dataset_std": [5.358466],
                "audio_length": [512], "noise": [False], "metrics": ['acc'],
                "loss": ['BCE'],
                "warmup": [True], "lrscheduler_start": [1], "lrscheduler_step": [1],
                "lrscheduler_decay": [0.85], "wa": [False], "wa_start": [1], "wa_end": [5]
                }

            pairs_path = [f'/home/almogk/FSL_TL_E_C/data_kfold/FSL_SETS/5w_1s_shot/train/{k}/2000/10000_train_2000_1C_1PC_task_sets.json',
                        f'/home/almogk/FSL_TL_E_C/data_kfold/FSL_SETS/5w_1s_shot/test/{k}/2000/10000_test_2000_1C_1PC_task_sets.json',
                        f'/home/almogk/FSL_TL_E_C/data_kfold/FSL_SETS/5w_1s_shot/val/{k}/120/600_val_120_1C_1PC_task_sets.json']
            
            ft_model_dir_pattern = '/home/almogk/FSL_TL_E_C/ast_class_ex_kfold/{}' + f'/{k}' + '/{}/models/best_audio_model.pth'
            
            # checkpoint_path = u.load_pt_ft_models_checkpoint_path(ft_model_dir_pattern) 
            checkpoint_path = [None]
            models_names = ['PT_(ImagNet)', 'PT_(ImagNet_AudioSet)'] 
            # models_names = ['scratch_T(ESC_35)',
                            # 'PT_(ImagNet)_FT_(ESC_35)',
                            # 'PT_(ImagNet_AudioSet)_FT_(ESC_35)']
            
            for mod_index, model in enumerate(checkpoint_path):
                
                if model == None:
                    param_combinations0 = list(ParameterGrid(parser_param))
                    param_combinations1 = list(ParameterGrid(parser_param1))
                    param_combinations = param_combinations0
                else:
                    param_combinations = list(ParameterGrid(parser_param1))
                
                for i, params in enumerate(param_combinations):
                    if model == None:
                        print(f'\nTraining model {i} from {i + 1}/{len(param_combinations)} with parameters:', params)
                    else:
                        print(f'\nTraining model {i} from {i + 1}/{len(param_combinations)} with parameters:', params)
                
                    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                    parser.set_defaults(**params)
                    args = parser.parse_args()
                    
                    train_loader, val_loader, test_loader = preper_data_for_sieamis_ast_model_train_test_val(args, pairs_path, batch_size)
                    
                    # Initialize the Siamese network with the given hyperparameters
                    audio_model = ast_mo.Siamese_ASTModel(input_tdim=args.audio_length,
                                                        con_los=False, fc=args.fc, fin=args.fin,
                                                        imagenet_pretrain=args.imagenet_pretrain,
                                                        audioset_pretrain=args.audioset_pretrain,
                                                        checkpoint_path=model)
                    start_time = time.time()
                    audio_model, optimizer, classifier = run_sieamis_ast(train_loader, val_loader, audio_model, args, models_names, i)
                    print('finished TRAIN at:')
                    print(f'{time.time() - start_time} sec\t {(time.time() - start_time)/60} min\n')
                    
                    # Test the Siamese network using the infer function
                    start_time = time.time()
                    test_metrics = sieamis_ast_infer(audio_model, test_loader, classifier)
                    # [x_d, y_d, cla_pred]
                    classi_x, classi_y, classi_pred = test_metrics[5]
                    
                    cla_bala_acc = balanced_accuracy_score(classi_y, classi_pred)
                    cla_acc = accuracy_score(classi_y, classi_pred)
                    
                    print('classifier scors: ')
                    print('acc scors: ', cla_acc)
                    print('balanc acc scors: ', cla_bala_acc)
                    
                    end_test_time = time.time()
                    print('finished test at:')
                    print(f'{end_test_time - start_time} sec\t {(end_test_time - start_time)/60} min\n')

                    # Record the metrics for this model
                    model_metrics = {
                        'stats': test_metrics[0],
                    'target': [i[0] for i in test_metrics[1].detach().cpu().numpy().tolist()],
                        'A_predictions': [i.detach().cpu().numpy().tolist() for i in test_metrics[3]],
                        'A_real_class': test_metrics[4]
                    }
                
                    # Save the metrics for this model to a file
                    with open(exp_dir + f'/{models_names[i]}/model_test_metrics.json', 'w') as f:
                        json.dump(model_metrics, f, indent=1)
                    
                    print(f'Training complete for model {i+1}/{len(param_combinations)} with parameters:', params)
    
    CHACK_SIEAMIS_TRAIN = False
    if CHACK_SIEAMIS_TRAIN:
        inclod_val = False
        models_names = ['scratch_T(ESC_35)', 
                        'PT_(ImagNet)', 'PT_(ImagNet)_FT_(ESC_35)',
                        'PT_(ImagNet_AudioSet)', 'PT_(ImagNet_AudioSet)_FT_(ESC_35)']
        
                
        sieamis_b_acc_test_kfold = []
        sieamis_mc_test_kfold = []
        
        sieamis_exp_PLOT_path = f"/home/almogk/FSL_TL_E_C/data_kfold/FSL_SETS/5w_1s_shot/test"
        
        for k in range(3):
            sieamis_exp_path = f"/home/almogk/FSL_TL_E_C/sieamis_ast_exp_kfold_2000/{k}/"
            
            sieamis_b_acc_test = []
            sieamis_mc_test = []
            
            sieamis_b_acc_val = []
            sieamis_mc_val = []
            for m_index, model in enumerate(range(len(models_names))):
                with open(sieamis_exp_path + f'{models_names[m_index]}/model_test_metrics.json', 'r') as f:
                    test_result = json.load(f)
                
                A_predictions_test = [ii for i in test_result['A_predictions'] for ii in i]
                
                binary_test, multi_test = cc.evaluate_sieamis_classification(A_predictions_test, test_result['target'], test_result['A_real_class'])
                sieamis_b_acc_test.append(binary_test[0])
                sieamis_mc_test.append(multi_test[0])
                
                if inclod_val:
                    val_result_csv = pd.read_csv(sieamis_exp_path + f'{models_names[m_index]}/val_result.csv')
                    sieamis_b_acc_val.append(val_result_csv.iloc[:, 0].max())
                    with open(sieamis_exp_path + f'{models_names[m_index]}/data_predict_val.json', 'r') as f:
                        val_result_json = json.load(f)
                    
                    binary_ground_truth_val = [sublist[0] for sublist in val_result_json['target']]
                    binary_val, multi_val = cc.evaluate_sieamis_classification(val_result_json['A_predictions'], binary_ground_truth_val, val_result_json['A_real_class'])
                    sieamis_mc_val.append([binary_val, multi_val])
            sieamis_b_acc_test_kfold.append(sieamis_b_acc_test)
            sieamis_mc_test_kfold.append(sieamis_mc_test)
        
        sieamis_b_acc_test_kfold_array = np.array(sieamis_b_acc_test_kfold)
        B_bala_acc_average_list, B_bala_acc_std_list = np.mean(sieamis_b_acc_test_kfold_array, axis=0).tolist(),  np.std(sieamis_b_acc_test_kfold_array, axis=0).tolist()
        
        sieamis_mc_test_kfold_array = np.array(sieamis_mc_test_kfold)
        MC_bala_acc_average_list, MC_bala_acc_std_list = np.mean(sieamis_mc_test_kfold_array, axis=0).tolist(),  np.std(sieamis_mc_test_kfold_array, axis=0).tolist()
        
        cc.plot_sieamis_mean_std_scors('Binary', models_names, [B_bala_acc_average_list, B_bala_acc_std_list], sieamis_exp_PLOT_path+f'/SIEAMIS_test_scors_B_2000.png')
        cc.plot_sieamis_mean_std_scors('Multi-class', models_names, [MC_bala_acc_average_list, MC_bala_acc_std_list], sieamis_exp_PLOT_path + f'/SIEAMIS_test_scors_MC_5_2000.png')

        if inclod_val: 
            cc.plot_sieamis_mean_std_scors('Binary', models_names, sieamis_b_acc_val, sieamis_exp_path+f'/val_scors_b.png')
            cc.plot_sieamis_scors(models_names, sieamis_mc_val, sieamis_exp_path + f'/val_scors_mc.png')
                                    
if __name__ == '__main__':
    main()
    