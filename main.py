# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from speechbrain.speechbrain.utils import Accuracy
# import matplotlib.pyplot as plt
# import commentjson as json
# from pathlib import Path
# import seaborn as sns
# import random, csv
# import itertools
# ######################## Installations ###############################
import os
import json
import time
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
import cosin_calc as cc
import infering as infer
import ast_models as ast_mo
from util import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import ParameterGrid
# #####################################################################

def sieamis_ast_infer(audio_model, test_loader):
    
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
    
    with torch.no_grad():
        for _, (audio_input1, audio_input2, labels, real_class) in enumerate(test_loader):
            
            audio_input1 = audio_input1.to(device, non_blocking=True)
            audio_input2 = audio_input2.to(device, non_blocking=True)

            # compute output
            audio_output = audio_model(audio_input1, audio_input2)
            
            predictions = audio_output.to('cpu').detach()
            predicted_thresholds = (predictions > 0.5).float()

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

        audio_output = torch.cat(A_predictions_tresholds)
        target = torch.cat(A_targets)
        A_predictions_tresholds = torch.cat(A_predictions_tresholds)
        
        stats = st.calculate_stats_(audio_output, target.to('cpu').detach())

    return stats, target, A_predictions_tresholds, A_predictions, A_real_class

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

def validate_sieamis(audio_model, loader, device, loss_fun):
        
        batch_time = AverageMeter()
        
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        
        audio_model = audio_model.to(device, non_blocking=True)
        
        # switch to evaluate mode
        audio_model.eval()

        end = time.time()
        A_predictions = []
        A_predictions_tresholds = []
        
        A_targets = []
        A_loss = []
        A_real_class = []
        
        with torch.no_grad():
            for _, (audio_input1, audio_input2, labels, real_class) in enumerate(loader):
                
                audio_input1 = audio_input1.to(device, non_blocking=True)
                audio_input2 = audio_input2.to(device, non_blocking=True)

                # compute output
                audio_output = audio_model(audio_input1, audio_input2)
                
                predictions = audio_output.to('cpu').detach()
                predicted_thresholds = (predictions > 0.5).float()

                # compute the loss
                labels = labels.view(labels.shape[0], -1)
                labels = labels.to(torch.float32)
                labels = labels.to(device, non_blocking=True)
                
                loss = loss_fun(audio_output.half(), labels.half())
                
                A_targets.append(labels.to('cpu').detach())
                A_real_class.append(real_class)
                A_predictions.append(predictions)
                A_predictions_tresholds.append(predicted_thresholds)
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()

            audio_output_ = torch.cat(A_predictions_tresholds)
            audio_output = torch.cat(A_predictions)
            
            target = torch.cat(A_targets)
            loss = np.mean(A_loss)
            
            stats = st.calculate_stats_(audio_output_, target)

        return stats, loss, target, audio_output_, A_real_class, audio_output
    
def super_vi_ast_cross_val(epochs, batch_size, cv_folds, class_lookup):
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_train_data_{fold_num}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_eval_data_{fold_num}.json", 
            "data_eval": "",
            "label_csv": "/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_exp/000/{fold_num}",
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
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_exp/10/{fold_num}",
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
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
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
                audio_output = audio_model(audio_input)
                # audio_output = torch.sigmoid(audio_output)
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

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))


        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

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
    train_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[0], audio_conf=train_audio_conf), 
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading train_loader\n')
    
    test_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[1], audio_conf=test_audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading test_loader\n')

    val_loader = torch.utils.data.DataLoader(
        dataloader.ESC_FSL_Dataset(pair_path[2], audio_conf=val_audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('finish loading val_loader\n')

    return train_loader, val_loader, test_loader

def run_sieamis_ast(train_loader, val_loader, audio_model, args, models_names, mod_index, ind, cont=False):
    
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
    
    global_step, epoch = 0, 0
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

        for i, (audio_input1, audio_input2, labels, _) in enumerate(train_loader):
            
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

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss.backward()
            # optimizer.step()

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
            
        end_epoch__time1 = time.time()
        print('finished training  epoch at:')
        print(f'{end_epoch__time1 - begin_time} sec\t {(end_epoch__time1 - begin_time)/60} min\n')
        
        
        print('start validation')
        stats, valid_loss, target, A_predictions_tresholds, A_real_class, A_predictions = validate_sieamis(audio_model, val_loader, device, loss_fn)
        
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
        
        epoch += 1
        scheduler.step()
        data_time.reset()
        batch_time.reset()
        loss_meter.reset()
        per_sample_time.reset()
        per_sample_dnn_time.reset()
        per_sample_data_time.reset()
    
    with open(exp_dir + f'/{models_names[mod_index]}/data_predict_val.json', 'w') as f:
        json.dump({'target': target.detach().cpu().numpy().tolist(),
                    'A_predictions': [i.detach().cpu().numpy().tolist() for i in A_predictions],
                    'A_real_class': A_real_class}, f, indent=1)
            
    print(f"\nfinished all {args.n_epochs} training...")
    print('training time: {:.3f}'.format(time.time()-start_time))
    
    return audio_model, optimizer

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
    
    class_lookup_35 = {'1': 0, '2': 1, '4': 2, '5': 3, '7': 4, '8': 5, '10': 6, '11': 7, '13': 8, 
                    '14': 9, '15': 10, '16': 11, '17': 12, '18': 13, '20': 14, '22': 15,
                    '25': 16, '26': 17, '27': 18, '28': 19, '29': 20, '30': 21, '31': 22,
                    '32': 23, '33': 24, '34': 25, '35': 26, '38': 27, '40': 28, '41': 29,
                    '42': 30, '43': 31, '45': 32, '48': 33, '49': 34}
    class_lookup_10 = {'0': 0, '3': 1, '9': 2, '19': 3, '21': 4, '36': 5, '39': 6, '44': 7, '46': 8, '47': 9}
    class_lookup_5 = {'6': 0, '12': 1, '23': 2, '24': 3, '37': 4}
    
   
    epochs = 15
    batch_size = 20
    cv_folds = 1
    
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
    train_support_set_num = [5000, 50000]
    test_support_set_num = [1000, 5000, 15000]
    val_support_set_num = 120
    
    make_task_sets_from_q = False
    if make_task_sets_from_q:
        
       u.make_task_sets_from_q(k_way, n_shot, train_support_set_num, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num, val_support_set_num)
    
    make_task_sets_from_ss = False
    if make_task_sets_from_ss:
        
        support_set_path = '/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/15000_test_suppotr_sets.json'
        u.make_task_sets_from_ss(support_set_path, k_way, n_shot, CODE_REPO_PATH, test_support_set_num)
    
    make_task_sets_from_unknown = False
    if make_task_sets_from_unknown:    
        u.make_task_sets_from_unknown_q(k_way, n_shot, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num)
    
    output_json = CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/test/embeddings___test_output.json'
    ft_model_dir_pattern = "/home/almogk/FSL_TL_E_C/ast_class_exp/{}/{}/models/best_audio_model.pth"
    audio_samples_json = "/home/almogk/FSL_TL_E_C/data/test_datafile/esc_fsl_test_data.json"
    
    EXTRACT_EMBEDDINGS = False
    if EXTRACT_EMBEDDINGS:
        
        audio_samples = infer.load_audio_samples(audio_samples_json)   
        
        audio_model_FF = infer.load_ast_tl_no_ft_model(512, False, False)
        audio_model_TF = infer.load_ast_tl_no_ft_model(512, True, False)
        audio_model_TT = infer.load_ast_tl_no_ft_model(512, True, True)
        embeddings_no_ft = infer.extract_embeddings([audio_model_FF, audio_model_TF, audio_model_TT], audio_samples, 'no_FT')
        
        models = infer.load_pt_ft_models(ft_model_dir_pattern, input_tdim=512)
        embeddings = infer.extract_embeddings(models, audio_samples, 'FT')
        
        embeddings_full = u.merge_dictionaries(embeddings_no_ft, embeddings)   
        infer.save_embeddings(embeddings_full, output_json)
    else:     
        
        with open(output_json, 'r') as f:
            embeddings_full = json.load(f)
            embeddings_full = {key: np.array(value) for key, value in embeddings_full.items()}
    
    
    models_names = ['scratch rand pos', 'scratch T(ESC-35)', 
                    'PT (ImagNet)', 'PT (ImagNet) FT (ESC-35)',
                    'PT (ImagNet, AudioSet)', 'PT (ImagNet, AudioSet) FT (ESC-35)']
    
    INFER = False
    if INFER:
        
        # pair_dic = u.make_all_pairs(CODE_REPO_PATH, test_support_set_num, query_c_num, query_c_size, k_way)
        save_file_path = '/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/'
        
        balanced_t = []
        with open(save_file_path + f'/75000_test_15000__1C_1PC_task_sets.json', 'r') as f:
            pairs = json.load(f)

        cosine_distances = cc.calculate_cosine_distances(embeddings_full, pairs)
        
        # perso_ss_tresholds = u.make_perso_ss_tresholds_openset(cosine_distances)
        # perso_ss_tresholds = u.make_perso_ss_tresholds(cosine_distances)
        
        with open('/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/ss_personal_tresh_no_q.json', 'r') as f:
            perso_ss_tresholds_no_q = json.load(f)
       
        with open('/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/ss_personal_tresh.json', 'r') as f:
            perso_ss_tresholds = json.load(f)
        
        tresh_sig_const = [0.1, 0.3, 0.5, 1, 1.2, 1.3, 1.5, 2]
        for sig_ind, sig in enumerate(tresh_sig_const):
        
            ss_tresholds_no_q = []
            ss_tresholds_no_q_max = []
            ss_tresholds_all = []
            ss_tresholds_all_0 = []
            for i in range(len(perso_ss_tresholds_no_q['mean_all'])):
                
                ss_tresholds_no_q.append([mean[0] + sig*std[0] for mean, std in zip(perso_ss_tresholds_no_q['mean_all'][i], perso_ss_tresholds_no_q['std_all'][i])])
                ss_tresholds_all.append([mean + sig*std for mean, std in zip(perso_ss_tresholds['mean_all'][i], perso_ss_tresholds['std_all'][i])])
                ss_tresholds_all_0.append([mean + sig*std for mean, std in zip(perso_ss_tresholds['mean_0'][i], perso_ss_tresholds['std_0'][i])])
                ss_tresholds_no_q_max.append(perso_ss_tresholds_no_q['max'][i])

            bc_p, accuracies_b, reports_b, conf_matrices_b = cc.evaluate_classification_per_ss(cosine_distances, pairs, ss_tresholds_all)
            _, accuracies_b_0, reports_b_0, conf_matrices_b_0 = cc.evaluate_classification_per_ss(cosine_distances, pairs, ss_tresholds_all_0)
            _, accuracies_b_per_ss_, _, _ = cc.evaluate_classification_per_ss(cosine_distances, pairs, ss_tresholds_no_q)
            _, accuracies_b_per_ss_nax, _, _ = cc.evaluate_classification_per_ss(cosine_distances, pairs, ss_tresholds_no_q_max)
            
            # # # bc_p, accuracies_b, reports_b, conf_matrices_b = cc.evaluate_classification_per_ss_balanc(cosine_distances, pairs, perso_ss_tresholds['threshold'])
            
            cc.plot_ss_scors(accuracies_b, accuracies_b_0, accuracies_b_per_ss_, accuracies_b_per_ss_nax, models_names, save_file_path+f'/scors_tresholds_{sig}.png')
            
        # cc.plot_scors(models_names, accuracies_b_per_ss, save_file_path+f'/scors_ps.png')
        
        b_a_t = []
        # define the start, end, and step values
        start = 0.3
        end = 0.8
        step = 0.1
        thresholds = [round(i * step + start, 2) for i in range(int((end - start) / step) + 1)]
        for t_ind, threshold in enumerate(thresholds):
            mc_p, accuracies_mc, reports_mc, conf_matrices_mc, _, bc_predictions, accuracies_b, reports_b, conf_matrices_b = cc.evaluate_classification(cosine_distances, pairs, threshold)
            
            cc.plot_scors(models_names, reports_mc, save_file_path+f'/scors_mc.png')
            
            # data_eval = {
            # 'multiclass pred': mc_p,
            # 'm_acc': accuracies_mc,
            # 'report_m': reports_mc,
            # 'binary pred': bc_predictions,
            # 'b_acc': accuracies_b,
            # 'report_b': reports_b
            # # 'con_m': [list(a) for a in conf_matrices_mc[0]],
            # # 'con_b': [list(a) for a in conf_matrices_b[0]],
            # }
            
            # data_eval = {'threshold': threshold,
            # 'binary pred': bc_predictions,
            # 'b_acc': accuracies_b,
            # 'report_b': reports_b
            # }
            
            # file_path = f'/data_eval_{threshold}.json'
            # # write the dictionary to a JSON file
            # with open(save_file_path + file_path, 'w') as f:
            #     json.dump(data_eval, f)
                
            # cc.calculate_statistics(cosine_distances, pairs, embeddings_full, save_file_path+'/stats.csv')
            # cc.plot_scors(reports_mc, models_names, accuracies_b, accuracies_mc, save_file_path+f'/scors_{threshold}.png')
            
            # for ii in range(len(models_names)):
            #     print(f"{models_names[ii]}")
            #     print(f"\n binary Accuracy: {accuracies_b[ii]}")
            #     print(f"\n binary Confusion Matrix:\n{conf_matrices_b[ii]}")
            #     print(f"\n binary Classification Report for:\n{reports_b[ii]}")
                
            #     print(f"\n Multiclass Accuracy: {accuracies_mc[ii]}")
            #     print(f"\n Multiclass Confusion Matrix:\n{conf_matrices_mc[ii]}")
            #     print(f"\n Multiclass Classification Report for:\n{reports_mc[ii]}")
                
            #     cc.plot_combined(conf_matrices_b[ii], reports_b[ii], 
            #                      models_names[ii], save_file_path+f'/{models_names[ii]}_cosine_binary_plot.png')
            balanced_t.append(accuracies_b)
        # cc.plot_b_scors(balanced_t, accuracies_b_per_ss_, accuracies_b_per_ss, thresholds, models_names, save_file_path+f'/scors_nbw_4_5_.png')
            
    
    SIEAMISE = True
    epochs = 15
    batch_size = 5
    exp_dir = ["/home/almogk/FSL_TL_E_C/sieamis_ast_exp_"]
    
    if SIEAMISE:
        # [1e-3, 1e-4, 1e-5, 1e-6]
        parser_param = {
            "n_class": [35], "model": ["ast"],
            'fc': ['m'], 'imagenet_pretrain': [True],'audioset_pretrain': [False, True],
            "dataset": ["esc50"], "exp_dir": exp_dir,
            "lr": [1e-4], "optim": ["adam"], "batch_size": [batch_size],
            "num_workers": [8], "n_epochs": [epochs], "lr_patience": [2],
            "n_print_steps": [1000], "save_model": [False], "freqm": [24],
            "timem": [96], "mixup": [0], "bal": [None], "fstride": [10],
            "tstride": [10],'fin': [False],
            "dataset_mean": [-6.6268077], "dataset_std": [5.358466],
            "audio_length": [512], "noise": [False], "metrics": ['acc'],
            "loss": ['BCE'],
            "warmup": [True], "lrscheduler_start": [1], "lrscheduler_step": [1],
            "lrscheduler_decay": [0.87], "wa": [False], "wa_start": [1], "wa_end": [5]
            }
        parser_param1 = {
            "n_class": [35], "model": ["ast"],
            'fc': ['m'], 'imagenet_pretrain': [False],'audioset_pretrain': [False],
            "dataset": ["esc50"], "exp_dir": exp_dir,
            "lr": [1e-4], "optim": ["adam"], "batch_size": [batch_size],
            "num_workers": [8], "n_epochs": [epochs], "lr_patience": [2],
            "n_print_steps": [1000], "save_model": [False], "freqm": [24],
            "timem": [96], "mixup": [0], "bal": [None], "fstride": [10],
            "tstride": [10], 'fin': [True],
            "dataset_mean": [-6.6268077], "dataset_std": [5.358466],
            "audio_length": [512], "noise": [False], "metrics": ['acc'],
            "loss": ['BCE'],
            "warmup": [True], "lrscheduler_start": [1], "lrscheduler_step": [1],
            "lrscheduler_decay": [0.87], "wa": [False], "wa_start": [1], "wa_end": [5]
            }

        pairs_path = ['/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/train/5000/25000_train_5000__1C_1PC_task_sets.json',
                  '/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/75000_test_15000__1C_1PC_task_sets.json',
                  '/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/val/120/3000_val_120__5C_1PC_task_sets.json']
        
        checkpoint_path = u.load_pt_ft_models_checkpoint_path(ft_model_dir_pattern) 
        # checkpoint_path.extend([None])
        # checkpoint_path = [None]
        # models_names = ['scratch_rand_pos', 'PT_(ImagNet)', 'PT_(ImagNet_AudioSet)'] 
        models_names = ['scratch_T(ESC_35)', 
                        'PT_(ImagNet)_FT_(ESC_35)', 
                        'PT_(ImagNet_AudioSet)_FT_(ESC_35)']
    
        for mod_index, model in enumerate(checkpoint_path):
            
            if model == None:
                param_combinations0 = list(ParameterGrid(parser_param))
                param_combinations1 = list(ParameterGrid(parser_param1))
                param_combinations = param_combinations1 + param_combinations0 
            else:
                param_combinations = list(ParameterGrid(parser_param1))
            
            for i, params in enumerate(param_combinations):
                if model == None:
                    print(f'\nTraining model {mod_index} from {i + 1}/{len(param_combinations)} with parameters:', params)
                else:
                    print(f'\nTraining model {mod_index} from {i + 1}/{len(param_combinations)} with parameters:', params)
            
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
                
                audio_model, optimizer = run_sieamis_ast(train_loader, val_loader, audio_model, args, models_names, i, i)
        
                # Test the Siamese network using the infer function
                test_metrics = sieamis_ast_infer(audio_model, test_loader)
                
                exp_dir_ = exp_dir + f'/{models_names[i]}'
                torch.save(audio_model.state_dict(), "%s/sieamis_audio_model.pth" % (exp_dir_))
                torch.save(optimizer.state_dict(), "%s/sieamis_optim_state.pth" % (exp_dir_))

                
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
        
        models_names = ['scratch_rand_pos_0', 'scratch_T(ESC_35)_0', 
                        'PT_(ImagNet)_1', 'PT_(ImagNet)_FT_(ESC_35)_0',
                        'PT_(ImagNet_AudioSet)_2', 'PT_(ImagNet_AudioSet)_FT_(ESC_35)_0']
        
        sieamis_exp_path = '/home/almogk/FSL_TL_E_C/sieamis_ast_exp/'
        
        sieamis_b_acc_val = []
        sieamis_mc_val = []
        sieamis_b_acc_test = []
        sieamis_mc_test = []
        
        for m_index, model in enumerate(range(len(models_names))):
            
            val_result_csv = pd.read_csv(sieamis_exp_path + f'{models_names[m_index]}/val_result.csv')
            sieamis_b_acc_val.append(val_result_csv.iloc[:, 0].max())
            
            with open(sieamis_exp_path + f'{models_names[m_index]}/data_predict_val.json', 'r') as f:
                val_result_json = json.load(f)
            
            with open(sieamis_exp_path + f'{models_names[m_index]}/model_test_metrics_{str(int(models_names[m_index].split("_")[-1])+1)}.json', 'r') as f:
                test_result = json.load(f)
            sieamis_b_acc_test.append(test_result['stats']['acc'])
            
            binary_ground_truth_val = [sublist[0] for sublist in val_result_json['target']]
            A_predictions_test = [ii for i in test_result['A_predictions'] for ii in i]
            
            binary_val, multi_val = cc.evaluate_sieamis_classification(val_result_json['A_predictions'], binary_ground_truth_val, val_result_json['A_real_class'])
            binary_test, multi_test = cc.evaluate_sieamis_classification(A_predictions_test, test_result['target'], test_result['A_real_class'])
            
            sieamis_mc_val.append([binary_val, multi_val])
            sieamis_mc_test.append([binary_test, multi_test])
        
        cc.plot_sieamis_b_scors(models_names, sieamis_b_acc_val, sieamis_exp_path+f'/val_scors_b.png')
        cc.plot_sieamis_b_scors(models_names, sieamis_b_acc_test, sieamis_exp_path+f'/test_scors_b.png')
        
        cc.plot_sieamis_scors(models_names, sieamis_mc_val, sieamis_exp_path + f'/val_scors_mc.png')
        cc.plot_sieamis_scors(models_names, sieamis_mc_test, sieamis_exp_path + f'/test_scors_mc.png')
                                    
if __name__ == '__main__':
    main()
    