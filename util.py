# import math
# import torch
# import random
# import pickle
# import torch.nn as nn

import os
import json
import itertools
import data_prep
import numpy as np
from scipy import stats
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter #, namedtuple

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

def read_json(path):
    
    with open(path, 'r') as f:
        json_obj = json.load(f)
    
    return json_obj

def write_json(path, obj):
    
    with open(path, 'w') as f:
        json.dump(obj, f)
    
def make_task_sets_from_q(k_way, n_shot, train_support_set_num, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num, val_support_set_num):
    
    for kway in range(len(k_way)):
        for nshot in range(len(n_shot)):
            
            for i in range(len(train_support_set_num)):
                for j in range(len(query_c_num)):
                    for k in range(len(query_c_size)):
                    
                        creat_f_s_sets(CODE_REPO_PATH, n_shot[nshot], k_way[kway], query_c_num[j], query_c_size[k], train_support_set_num[i], test_support_set_num[i], val_support_set_num, i)
                        print(f"finished creating task sets for {n_shot[nshot]} shot {k_way[kway]} way {query_c_num[j]} query classes {query_c_size[k]} query examples per class")
    
    print('finished all tasks')

def make_task_sets_from_unknown_q(k_way, n_shot, query_c_num, query_c_size, CODE_REPO_PATH, test_support_set_num, text, data_path):
    
    for kway in range(len(k_way)):
        for nshot in range(len(n_shot)):
            
            for i in range(len(test_support_set_num)):
                for j in range(len(query_c_num)):
                    for k in range(len(query_c_size)):
                    
                        creat_unknown_f_s_sets(CODE_REPO_PATH, n_shot[nshot], k_way[kway], query_c_num[j], query_c_size[k], test_support_set_num[i], data_path, text)
                        print(f"finished creating known/unknown task sets for {n_shot[nshot]} shot {k_way[kway]} way {query_c_num[j]} query classes {query_c_size[k]} query examples per class")
    
    print('finished all tasks')

def load_pt_ft_models_checkpoint_path(model_dir_pattern):
            
    models_checkpoint_path = []
    for j in ["00", "10", "11"]:
        
        model_path = model_dir_pattern.format(j, 2)
        if os.path.exists(model_path):
            models_checkpoint_path.append(model_path)
    
    return models_checkpoint_path

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

def create_unknown_q_sets(json_data_path, n_shot, k_way, support_set_num, s, query_c_num, query_c_size, save_path):
    
    # load the input JSON file
    with open(json_data_path, 'r') as f:
        input_dict = json.load(f)
    
    # extract the data array from the input dictionary
    data = input_dict['data']
    
    SAVE_PATH = save_path + f'/{s}/{support_set_num}'
    if os.path.exists(SAVE_PATH) == False:
        os.mkdir(SAVE_PATH)
        
    # write the output JSON file
    with open(SAVE_PATH + f'/{support_set_num}_{s}_suppotr_sets.json', 'r') as f:
        support_set_dict = json.load(f)
    
    q_dict = data_prep.create_open_set_queries(data, SAVE_PATH + f'/{support_set_num}_{s}_suppotr_sets.json', query_c_num, query_c_size, v_o_t=s)
    
    # write the output JSON file
    with open(SAVE_PATH + f'/{len(q_dict)}_{s}_{query_c_num}C_{query_c_size}PC_q_openset.json', 'w') as f:
        json.dump(q_dict, f)
    
    return support_set_dict, q_dict

def create_task_sets(support_set_dict, q_dict, support_set_num, query_c_num, query_c_size, s, save_path, n_shot, openset=False):
    
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
    if openset:
        with open(SAVE_PATH + f'/{len(task_sets)}_{s}_{support_set_num}_{query_c_num}C_{query_c_size}PC_task_sets_openset.json', 'w') as f:
            json.dump(task_sets, f)
    else:        
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

def creat_unknown_f_s_sets(CODE_REPO_PATH, n_shot, k_way, query_c_num, query_c_size, test_support_set_num, data_path, text):
   
    SAVE_PATH = f'/home/almogk/FSL_TL_E_C/data/FSL_SETS/{k_way}w_{n_shot}s_shot'
    if os.path.exists(SAVE_PATH) == False:
        os.mkdir(SAVE_PATH)
        os.mkdir(SAVE_PATH + '/train')
        os.mkdir(SAVE_PATH + '/test')
        os.mkdir(SAVE_PATH + '/val')

    # create the query sets
    test_support_set_dict, test_q_dict = create_unknown_q_sets(CODE_REPO_PATH + data_path, 
                        n_shot, k_way, test_support_set_num, text, query_c_num, query_c_size, SAVE_PATH)
    
    test_task_set = create_task_sets(test_support_set_dict, test_q_dict, 
                                    test_support_set_num, query_c_num, query_c_size, text, SAVE_PATH, n_shot, True)
    
def merge_dictionaries(dict1, dict2):
   
    merged_dict = {}
    for key in dict1.keys(): # [0] is the scratch, [1] is the ImagNet, [2] is the ImagNet + AudioSet
            merged_dict[key] = [dict1[key][0], dict2[key][0], dict1[key][1], dict2[key][1], dict1[key][2], dict2[key][2]]

    return merged_dict

def create_task_of_ss_test_sets(support_set_dict, support_set_num, s, save_path, n_shot):

    task_sets = []
    SAVE_PATH = save_path + f'/data/FSL_SETS/5w_1s_shot/{s}/{support_set_num}'
    # Iterate through all support sets
    for i in range(support_set_num):
        
        ss_labels = support_set_dict['class_names'][i]
        ss_wavs = support_set_dict['class_roots'][i]

        # Create pairs of samples from the current support set
        pairs_wav = list(itertools.combinations(ss_wavs, 2))
        pairs_lable = list(itertools.combinations(ss_labels, 2))
        
        for pair_index, pair in enumerate(pairs_wav):
            task_sets.append([0, pairs_lable[pair_index][0], pairs_lable[pair_index][1], pair[0], pair[1]])
    
    # write the output JSON file
    with open(SAVE_PATH + f'/{len(task_sets)}_{s}_{support_set_num}_ss_task_sets.json', 'w') as f:
        json.dump(task_sets, f)
    
    return task_sets

def make_task_sets_from_ss(support_set_path, k_way, n_shot, CODE_REPO_PATH, support_set_num, s):
    
    with open(support_set_path, 'r') as f:
        support_set_dic = json.load(f)
    
    sets = create_task_of_ss_test_sets(support_set_dic, support_set_num[-1], s, CODE_REPO_PATH, n_shot)

def make_perso_ss_param(cosine_distances, save_f):
    
    dis_ss = []
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
    
        mc_pair = []
        p_i = []
        b_max_pred = []
        for i, dis in enumerate(model_distances):
            mc_pair.append(dis)
            if (i+1) % 5 == 0:
                b_max_pred.append([mc_pair, mc_pair.index(max(mc_pair))])
                p_i.append(model_distances.index(max(mc_pair)))
                mc_pair = []
        dis_ss.append(b_max_pred)
        
    diff_all_M_list, std0_all_M_list, mean0_all_M_list, std_all_M_list, mean_all_M_list, min_all_M_list, max_all_M_list, max0_all_M_list, mad_all_M_list, mad0_all_M_list, precen_95_all_M_list, precen_95_0_all_M_list = [[] for _ in range(12)]
    for i, dist in enumerate(dis_ss): 

        diff_all_list, std0_all_list, mean0_all_list, std_all_list, mean_all_list, min_all_list, max_all_list, max0_all_list, mad_all_list, mad0_all_list, precen_95_all_list, precen_95_0_all_list = [[] for _ in range(12)]
        for _, (ss_dist, posi_ind) in enumerate(dist):
            
            ss_dist_wo_posi = np.delete(ss_dist, posi_ind)
            max_ind = np.argmax(ss_dist_wo_posi)
            second_max_val = ss_dist_wo_posi[max_ind]
            mean_ss = np.mean(ss_dist_wo_posi)
            std_ss = np.std(ss_dist_wo_posi)
            max_val = np.max(ss_dist)
            mad_val = stats.median_abs_deviation(ss_dist)
            mad0_val = stats.median_abs_deviation(ss_dist_wo_posi)
            precen_95 = np.percentile(ss_dist, 95)
            precen_0_95 = np.percentile(ss_dist_wo_posi, 95)
            
            precen_95_all_list.append(precen_95)
            precen_95_0_all_list.append(precen_0_95)
            mad_all_list.append(mad_val)
            mad0_all_list.append(mad0_val)
            max_all_list.append(max_val)
            min_all_list.append(np.min(ss_dist))
            mean_all_list.append(np.mean(ss_dist))
            std_all_list.append(np.std(ss_dist))
            max0_all_list.append(second_max_val)
            mean0_all_list.append(mean_ss)
            std0_all_list.append(std_ss)
            diff_all_list.append(ss_dist[posi_ind]- second_max_val)
        
        precen_95_all_M_list.append(precen_95_all_list)
        precen_95_0_all_M_list.append(precen_95_0_all_list)
        mad_all_M_list.append(mad_all_list)
        mad0_all_M_list.append(mad0_all_list)
        diff_all_M_list.append(diff_all_list)
        std0_all_M_list.append(std0_all_list) 
        mean0_all_M_list.append(mean0_all_list)
        std_all_M_list.append(std_all_list)
        mean_all_M_list.append(mean_all_list)
        min_all_M_list.append(min_all_list)
        max_all_M_list.append(max_all_list)
        max0_all_M_list.append(max0_all_list)
        
    perso_ss_tresholds = {'max': max_all_M_list,
                        'min': min_all_M_list,
                        'max0': max0_all_M_list,
                        'mean_all': mean_all_M_list,
                        'std_all': std_all_M_list,
                        'mean_0': mean0_all_M_list,
                        'std_0': std0_all_M_list,
                        'f_s_dif': diff_all_M_list, 
                        'MAD': mad_all_M_list,
                        'MAD_0': mad0_all_M_list,
                        'PERC_95': precen_95_all_M_list,
                        'PERC_95_0': precen_95_0_all_M_list}
    
    with open(save_f, 'w') as f:
        json.dump(perso_ss_tresholds, f)
        
    return perso_ss_tresholds

def make_perso_ss_param_no_q(cosine_distances, save_f):
    
    dis_ss = []
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        
        mc_pair = []
        b_max_pred = []
        for i, dis in enumerate(model_distances):
            mc_pair.append(dis)
            if (i+1) % 10 == 0:
                b_max_pred.append(mc_pair)
                mc_pair = []
        dis_ss.append(b_max_pred)
        
    std_all_M_list = []
    mean_all_M_list = []
    min_all_M_list = []
    max_all_M_list = []
    mad_all_M_list = []
    precen_95_all_M_list = []
    
    for i, dist in enumerate(dis_ss): 
        std_all_list = []
        mean_all_list = []
        min_all_list = []
        max_all_list = []
        mad_all_list = []
        precen_95_all_list = []
        
        for _, ss_dist in enumerate(dist):
            
            max_ind = np.argmax(ss_dist)
            min_ind = np.argmin(ss_dist)
            
            max_val = ss_dist[max_ind]
            min_val = ss_dist[min_ind]
            mean_ss = np.mean(ss_dist)
            std_ss = np.std(ss_dist)
            mad_val = stats.median_abs_deviation(ss_dist)
            precen_95 = np.percentile(ss_dist, 95)
            
            mean_ss_1 = np.mean(ss_dist[:4])
            std_ss_1 = np.std(ss_dist[:4])
            mad_val_1 = stats.median_abs_deviation(ss_dist[:4])
            precen_95_1 = np.percentile(ss_dist[:4], 95)
            max_val_1 = np.max(ss_dist[:4])
            min_val_1 = np.min(ss_dist[:4])

            mean_ss_2 = np.mean(ss_dist[:1]+ss_dist[4:7])
            std_ss_2 = np.std(ss_dist[:1]+ss_dist[4:7]) 
            mad_val_2 = stats.median_abs_deviation(ss_dist[:1]+ss_dist[4:7])
            precen_95_2 = np.percentile(ss_dist[:1]+ss_dist[4:7], 95)
            max_val_2 = np.max(ss_dist[:1]+ss_dist[4:7])
            min_val_2 = np.min(ss_dist[:1]+ss_dist[4:7])
            
            mean_ss_3 = np.mean(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9])
            std_ss_3 = np.std(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9])
            mad_val_3 = stats.median_abs_deviation(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9])
            precen_95_3 = np.percentile(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9], 95)
            max_val_3 = np.max(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9])
            min_val_3 = np.min(ss_dist[1:2]+ss_dist[4:5]+ss_dist[7:9])
        
            mean_ss_4 = np.mean(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:])
            std_ss_4 = np.std(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:])
            mad_val_4 = stats.median_abs_deviation(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:])
            precen_95_4 = np.percentile(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:], 95)
            max_val_4 = np.max(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:])
            min_val_4 = np.min(ss_dist[2:3]+ss_dist[5:6]+ss_dist[9:])
            
            mean_ss_5 = np.mean(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:])
            std_ss_5 = np.std(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:])
            mad_val_5 = stats.median_abs_deviation(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:])
            precen_95_5 = np.percentile(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:], 95)
            max_val_5 = np.max(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:])
            min_val_5 = np.min(ss_dist[2:3]+ss_dist[6:7]+ss_dist[8:])
            
            max_all_list.append([max_val, max_val_1, max_val_2, max_val_3, max_val_4, max_val_5])
            min_all_list.append([min_val, min_val_1, min_val_2, min_val_3, min_val_4, min_val_5])
            mad_all_list.append([mad_val, mad_val_1, mad_val_2, mad_val_3, mad_val_4, mad_val_5])
            precen_95_all_list.append([precen_95, precen_95_1, precen_95_2, precen_95_3, precen_95_4, precen_95_5])
            mean_all_list.append([mean_ss, mean_ss_1, mean_ss_2, mean_ss_3, mean_ss_4, mean_ss_5])
            std_all_list.append([std_ss, std_ss_1, std_ss_2, std_ss_3, std_ss_4, std_ss_5])
        
        std_all_M_list.append(std_all_list)
        mean_all_M_list.append(mean_all_list)
        min_all_M_list.append(min_all_list)
        max_all_M_list.append(max_all_list)
        mad_all_M_list.append(mad_all_list)
        precen_95_all_M_list.append(precen_95_all_list)
        
    perso_ss_tresholds = {'max': max_all_M_list,
                        'min': min_all_M_list,
                        'mean_all': mean_all_M_list,
                        'std_all': std_all_M_list, 
                        'MAD': mad_all_M_list,
                        'PRECEN_95': precen_95_all_M_list}
    
    with open(save_f, 'w') as f:
        json.dump(perso_ss_tresholds, f)
        
    return perso_ss_tresholds
