# ######################## Installations ###############################
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import pickle
import random
import os, csv
import datetime
import argparse
import data_prep
import dataloader
import cosin_calc as cc
import infering as infer
import ast_models as models
import stats as st
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch.nn as nn
from util import AverageMeter
import util as u
from torch.cuda.amp import autocast, GradScaler

# #####################################################################
def super_vi_ast_cross_val(epochs, batch_size, cv_folds, class_lookup):
    
    for fold_num in range(1, cv_folds+1):
        
        print('process fold number: ', fold_num)
        parser_param = {
            "data_train": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_train_data_{fold_num}.json", 
            "data_val": f"/home/almogk/FSL_TL_E_C/data/train_datafile/train_cla_datafile/35_esc_eval_data_{fold_num}.json", 
            "data_eval": "",
            "label_csv": "/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv", "n_class": 35, "model": "ast",
            "dataset": "esc50", "exp_dir": f"/home/almogk/FSL_TL_E_C/ast_class_exp/00/{fold_num}",
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

    def validate_ensemble(args, epoch):
        exp_dir = args.exp_dir
        target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
        if epoch == 1:
            cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
        else:
            cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
            predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
            cum_predictions = cum_predictions + predictions
            # remove the prediction file to save storage space
            os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

        cum_predictions = cum_predictions / epoch
        np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

        stats = st.calculate_stats(cum_predictions, target)
        return stats

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
    # for amp
    scaler = GradScaler()

    print("start training...")
    result = np.zeros([args.n_epochs, 10])
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
                audio_output = audio_model(audio_input)
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

        # ensemble results
        cum_stats = validate_ensemble(args, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        
        print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        # print("Avg Precision: {:.6f}".format(average_precision))
        # print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(st.d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

       
        result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, st.d_prime(mAUC), loss_meter.avg, valid_loss, cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
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

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

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
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, class_map=class_map),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, class_map=class_map),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')

    print("\nCreating experiment directory: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

    return train_loader, val_loader, audio_model, args


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
    
    data_prep.get_data_for_ast_model(ESC50_PATH, CODE_REPO_PATH, train_class_names, test_class_names, val_class_names, class_lookup_35)
    
    epochs = 20
    batch_size = 24
    cv_folds = 5
    
    Train_base_model = False

    if Train_base_model:
        super_vi_ast_cross_val(epochs, batch_size, cv_folds, class_lookup_35)

    # set the number of examples per class and number of classes
    n_shot = [1] # [1, 2, 3, 5]
    k_way = [5] # [2, 3, 5]
    
    query_c_num = [1, 3, 5] # number of query classes for each support set
    query_c_size = [1, 2, 4] # number of query examples per query class
    
    # set the sizes of the support sets
    train_support_set_num = [5000, 10000, 25000, 50000]
    test_support_set_num = [1000, 2000, 5000, 15000]
    val_support_set_num = 120
    make_task_sets = False
    
    if make_task_sets:
        
        for kway in range(len(k_way)):
            for nshot in range(len(n_shot)):
                
                for i in range(len(train_support_set_num)):
                    for j in range(len(query_c_num)):
                        for k in range(len(query_c_size)):
                        
                            u.creat_f_s_sets(CODE_REPO_PATH, n_shot[nshot], k_way[kway], query_c_num[j], query_c_size[k], train_support_set_num[i], test_support_set_num[i], val_support_set_num, i)
                            print(f"finished creating task sets for {n_shot[nshot]} shot {k_way[kway]} way {query_c_num[j]} query classes {query_c_size[k]} query examples per class")
    

    output_json = CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/test/embeddings_test_output.json'
    
    ft_model_dir_pattern = "/home/almogk/FSL_TL_E_C/ast_class_exp/{}/{}/models/best_audio_model.pth"
    
    audio_samples_json = "/home/almogk/FSL_TL_E_C/data/test_datafile/esc_fsl_test_data.json"
    
    EXTRACT_EMBEDDINGS = False
    
    if EXTRACT_EMBEDDINGS:
        
        audio_samples = infer.load_audio_samples(audio_samples_json)   
        
        models = infer.load_pt_ft_models(ft_model_dir_pattern, input_tdim=512)
        embeddings = infer.extract_embeddings(models, audio_samples, 'FT')
        
        audio_model_FF, audio_model_TF, audio_model_TT = infer.load_ast_tl_no_ft_model(512)
        embeddings_no_ft = infer.extract_embeddings([audio_model_FF, audio_model_TF, audio_model_TT], audio_samples, 'no_FT')
        
        embeddings_full = u.merge_dictionaries(embeddings_no_ft, embeddings)    
        infer.save_embeddings(embeddings_full, output_json)
    
    else:     
        with open(output_json, 'r') as f:
            embeddings_full = json.load(f)
            embeddings_full = {key: np.array(value) for key, value in embeddings_full.items()}
    
    
    pair_dic = u.make_all_pairs(CODE_REPO_PATH, test_support_set_num, query_c_num, query_c_size, k_way)
    
    with open('/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/375000_test_15000__5C_1PC_task_sets.json', 'r') as f:
        pairs = json.load(f)
    cosine_distances = cc.calculate_cosine_distances(embeddings_full, pairs)
    
    classifications_repo_mc, accuracies_b, accuracies_mc, _, _, _ = cc.evaluate_classification(cosine_distances, pairs)
    
    models_names = ['scratch', 'scratch + FT(ESC-35)', 
                    'PT(ImagNet)', 'PT(ImagNet) + FT(ESC-35)',
                    'PT(ImagNet + AudioSet)', 'PT(ImagNet + AudioSet) + FT(ESC-35)']

    path = '/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test/15000/5_scors.png'
    
    cc.plot_scors(classifications_repo_mc, models_names, accuracies_b, accuracies_mc, path)
        
    # for ii in range(len(models_names)):
        
    #     print(f"{models_names[ii]}")
    #     print(f"\n Accuracy: {accuracy[ii]}")
    #     print(f"\n Confusion Matrix:\n{conf_matrix[ii]}")
    #     print(f"\n Classification Report for:\n{report[ii]}")
            
    #     cc.plot_combined(conf_matrix[ii], report[ii], models_names[ii], f'/home/almogk/FSL_TL_E_C/data/cosin_sim/test/25000_5000_1_1/BC')


    
if __name__ == '__main__':
    main()
   