import json
import math
import os
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import data_prep
from collections import Counter, namedtuple

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_all_pairs(CODE_REPO_PATH, support_set_num, query_c_num, query_c_size, k_way):
    pairs_json_path = [] 
    pairs_dict = {}

    for ssn in support_set_num:
        for qc in query_c_num:
            for qcn in query_c_size:
                pairs_json_path.append(CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/test/{ssn}/{k_way[0]*ssn*qc*qcn}_test_{ssn}__{qc}C_{qcn}PC_task_sets.json')
                
                if os.path.exists(f'/home/almogk/FSL_TL_E_C/data/cosin_sim/test/{k_way[0]*ssn*qc*qcn}_{ssn}_{qc}_{qcn}') == False:
                    os.mkdir(f'/home/almogk/FSL_TL_E_C/data/cosin_sim/test/{k_way[0]*ssn*qc*qcn}_{ssn}_{qc}_{qcn}')
                    os.mkdir(f'/home/almogk/FSL_TL_E_C/data/cosin_sim/test/{k_way[0]*ssn*qc*qcn}_{ssn}_{qc}_{qcn}/BC')
                    os.mkdir(f'/home/almogk/FSL_TL_E_C/data/cosin_sim/test/{k_way[0]*ssn*qc*qcn}_{ssn}_{qc}_{qcn}/MLC')
                
                with open(CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/test/{ssn}/{k_way[0]*ssn*qc*qcn}_test_{ssn}__{qc}C_{qcn}PC_task_sets.json', 'r') as f:
                    pairs = json.load(f)
                    pairs_dict[f'{k_way[0]*ssn*qc*qcn}_{ssn}_{qc}C_{qcn}PC'] = pairs
    return pairs_dict

def calculate_label_counts(folder_path):
    
    result = {}
    
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and subfolder.name.isdigit():
            file_path = subfolder / f'{subfolder.name}_test_suppotr_sets.json'
            with open(file_path, 'r') as f:
                data = json.load(f)
            counter = Counter()
            for lst in data['class_names']:
                counter.update(lst)
            total_labelings = sum(counter.values())
            percentages = {label: (count / total_labelings) * 100 for label, count in counter.items()}
            result[file_path.name] = {'counts': counter, 'percentages': percentages}
    
    return result

def display_label_bar_chart(label_counts):
    all_counts = []
    all_percentages = []
    for file_counts in label_counts.values():
        counts = file_counts['counts']
        percentages = file_counts['percentages']
        all_counts.append(counts)
        all_percentages.append(percentages)
        print(f'Counts for {file_counts}:')
        for label, count in counts.items():
            print(f'{label}: {count}')
        print(f'Percentages for {file_counts}:')
        for label, percentage in percentages.items():
            print(f'{label}: {percentage:.2f}%')
    all_counts = sum(all_counts, Counter())
    all_percentages = {label: np.mean([p[label] for p in all_percentages]) for label in all_counts.keys()}
    # std_percentages = {label: np.std([p[label] for p in all_percentages]) for label in all_counts.keys()}
    labels, values = zip(*all_percentages.items())
    plt.bar(labels, values, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Percentage')
    plt.title('Label distribution')
    plt.savefig(os.path.join('/home/almogk/FSL_TL_E_C/data/FSL_SETS/5w_1s_shot/test', f'sss.png'), bbox_inches='tight')
    plt.show()

def create_support_sets_and_q(json_data_path, n_shot, k_way, support_set_num, s, query_c_num, query_c_size, save_path):
    
    # load the input JSON file
    with open(json_data_path, 'r') as f:
        input_dict = json.load(f)
    # extract the data array from the input dictionary
    
    data = input_dict['data']
    support_set_dict = data_prep.create_support_sets(data, n_shot, k_way, support_set_num)
    
    SAVE_PATH = save_path + f'/{s}/{support_set_num}'
    if os.path.exists(SAVE_PATH) == False:
        os.mkdir(SAVE_PATH)
        
    # write the output JSON file
    with open(SAVE_PATH + f'/{support_set_num}_{s}_suppotr_sets.json', 'w') as f:
        json.dump(support_set_dict, f)
        
    q_dict = data_prep.create_query_sets(data, SAVE_PATH + f'/{support_set_num}_{s}_suppotr_sets.json', query_c_num, query_c_size)
    
    # write the output JSON file
    with open(SAVE_PATH + f'/{len(q_dict)}_{s}_{query_c_num}C_{query_c_size}PC_q.json', 'w') as f:
        json.dump(q_dict, f)
    
    return support_set_dict, q_dict

def create_task_sets(support_set_dict, q_dict, support_set_num, query_c_num, query_c_size, s, save_path, n_shot):
    
    task_sets = []
    SAVE_PATH = save_path + f'/{s}/{support_set_num}'
    for i in range(support_set_num):
        for q_c in range(query_c_num):
            q_label = q_dict[i][q_c][0]
            q_wavs = q_dict[i][q_c][1]
            
            ss_labels = support_set_dict['class_names'][i]
            if n_shot > 1:
                new_dict = {}
                for i in range(len(ss_labels) * n_shot):
                    new_key = i
                    old_key = i//n_shot
                    new_dict[new_key] = ss_labels[old_key]
                ss_labels = new_dict
                
            ss_wavs = support_set_dict['class_roots'][i]
            
            for q_wav in q_wavs:
                for ii, ss_wav in enumerate(ss_wavs):
                    if ss_labels[ii] == q_label:
                        task_sets.append([1, q_label, ss_labels[ii], q_wav, ss_wav])
                    else:
                        task_sets.append([0, q_label, ss_labels[ii], q_wav, ss_wav])
    
    # write the output JSON file
    with open(SAVE_PATH + f'/{len(task_sets)}_{s}_{support_set_num}_{query_c_num}C_{query_c_size}PC_task_sets.json', 'w') as f:
        json.dump(task_sets, f)
        
    return task_sets

def creat_f_s_sets(CODE_REPO_PATH, n_shot, k_way, query_c_num, query_c_size, train_support_set_num, test_support_set_num, val_support_set_num, val_support_set):
   
    SAVE_PATH = f'/home/almogk/FSL_TL_E_C/data/FSL_SETS/{k_way}w_{n_shot}s_shot'
    if os.path.exists(SAVE_PATH) == False:
        os.mkdir(SAVE_PATH)
        os.mkdir(SAVE_PATH + '/train')
        os.mkdir(SAVE_PATH + '/test')
        os.mkdir(SAVE_PATH + '/val')

    # create the support sets and query sets
    train_support_set_dict, train_q_dict = create_support_sets_and_q(CODE_REPO_PATH + '/data/train_datafile/train_fsl_datafile/esc_fsl_train_data.json', 
                        n_shot, k_way, train_support_set_num, 'train', query_c_num, query_c_size, SAVE_PATH)
    
    test_support_set_dict, test_q_dict = create_support_sets_and_q(CODE_REPO_PATH + '/data/test_datafile/esc_fsl_test_data.json', 
                        n_shot, k_way, test_support_set_num, 'test', query_c_num, query_c_size, SAVE_PATH)
    
    if val_support_set == 0:
        val_support_set_dict, val_q_dict = create_support_sets_and_q(CODE_REPO_PATH + '/data/val_datafile/esc_fsl_val_data.json', 
                            n_shot, k_way, val_support_set_num, 'val', query_c_num, query_c_size, SAVE_PATH)
        
        val_task_set = create_task_sets(val_support_set_dict, val_q_dict, 
                                    val_support_set_num, query_c_num, query_c_size, 'val', SAVE_PATH, n_shot)
    
    # create the task sets
    train_task_set = create_task_sets(train_support_set_dict, train_q_dict, 
                                    train_support_set_num, query_c_num, query_c_size, 'train', SAVE_PATH, n_shot)
    
    test_task_set = create_task_sets(test_support_set_dict, test_q_dict, 
                                    test_support_set_num, query_c_num, query_c_size, 'test', SAVE_PATH, n_shot)

def merge_dictionaries(dict1, dict2):
   
    merged_dict = {}
    for key in dict1.keys(): # [0] is the scratch, [1] is the ImagNet, [2] is the ImagNet + AudioSet
            merged_dict[key] = [dict1[key][0], dict2[key][0], dict1[key][1], dict2[key][1], dict1[key][2], dict2[key][2]]

    return merged_dict




def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    S[i][j] is computed as the dot product between the meanpooled embeddings of
    the ith image output and jth audio output
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 4)
    n = image_outputs.size(0)
    imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
    audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_audio_outputs_list = []
    for idx in range(n):
        nF = max(1, nframes[idx])
        pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
    pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
    S = torch.mm(pooled_image_outputs, pooled_audio_outputs.t())
    return S

def one_imposter_index(i, N):
    imp_ind = random.randint(0, N - 2)
    if imp_ind == i:
        imp_ind = N - 1
    return imp_ind

def basic_get_imposter_indices(N):
    imposter_idc = []
    for i in range(N):
        # Select an imposter index for example i:
        imp_ind = one_imposter_index(i, N)
        imposter_idc.append(imp_ind)
    return imposter_idc

def semihardneg_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Impostors are taken
    to be the most similar point to the anchor that is still less similar to the anchor
    than the positive example.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    eps = 1e-12
    # All examples less similar than ground truth
    mask = (Sdiff < -eps).type(torch.LongTensor)
    maskf = mask.type_as(S)
    # Mask out all examples >= gt with minimum similarity
    Sp = maskf * Sdiff + (1 - maskf) * torch.min(Sdiff).detach()
    # Find the index maximum similar of the remaining
    _, idc = Sp.max(dim=1)
    idc = idc.data.cpu()
    # Vector mask: 1 iff there exists an example < gt
    has_neg = (mask.sum(dim=1) > 0).data.type(torch.LongTensor)
    # Random imposter indices
    random_imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # Use hardneg if there exists an example < gt, otherwise use random imposter
    imp_idc = has_neg * idc + (1 - has_neg) * random_imp_ind
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_idc):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

def sampled_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Imposters are
    randomly sampled from the columns of S.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_ind):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        print('current learing rate is {:f}'.format(lr))
    lr = cur_lr  * 0.1
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])



PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])

