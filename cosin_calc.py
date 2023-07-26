import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.spatial.distance import mahalanobis
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, roc_curve, balanced_accuracy_score #, average_precision_score, pairwise_distances  

def calculate_statistics(cosine_distances, pairs, embeddings, save_file_path):
    
    stats_labeled = {}
    stats_labeled_1 = {}
    stats_labeled_0 = {}
    stats_embeddings = {}
    
    # Calculate the statistics for each group
    for i in range(0, len(cosine_distances[0])):
        
        distances_labeled = [dist[i] for dist in cosine_distances]
        distances_labeled_1_i = [dist[i] for j, dist in enumerate(cosine_distances) if pairs[j][0] == 1]
        distances_labeled_0_i = [dist[i] for j, dist in enumerate(cosine_distances) if pairs[j][0] == 0]
        
        key = f'dist_{i}'
         
        stats_labeled[key] = {
            'mean': np.mean(distances_labeled),
            'std': np.std(distances_labeled),
            'min': np.min(distances_labeled),
            'max': np.max(distances_labeled),
            'median': np.median(distances_labeled),
            'q1': np.percentile(distances_labeled, 25),
            'q3': np.percentile(distances_labeled, 75)
        }
    
        stats_labeled_1[key] = {
            'mean': np.mean(distances_labeled_1_i),
            'std': np.std(distances_labeled_1_i),
            'min': np.min(distances_labeled_1_i),
            'max': np.max(distances_labeled_1_i),
            'median': np.median(distances_labeled_1_i),
            'q1': np.percentile(distances_labeled_1_i, 25),
            'q3': np.percentile(distances_labeled_1_i, 75)
        }
        
        stats_labeled_0[key] = {
            'mean': np.mean(distances_labeled_0_i),
            'std': np.std(distances_labeled_0_i),
            'min': np.min(distances_labeled_0_i),
            'max': np.max(distances_labeled_0_i),
            'median': np.median(distances_labeled_0_i),
            'q1': np.percentile(distances_labeled_0_i, 25),
            'q3': np.percentile(distances_labeled_0_i, 75)
        }
        
        emb_labeled_1 = []
        emb_labeled_0 = []
        for sample, emb_list in embeddings.items():
            emb_pairs = [(emb_list[j], emb_list[j+1]) for j in range(0, len(emb_list)-1, 2)]
            for j, emb_pair in enumerate(emb_pairs):
                emb_pair_mean = np.mean(np.array(emb_pair), axis=0)
                key_emb = f'{sample}_model_{j}_emb_{i}'
                if pairs[j*2][0] == 1 and pairs[j*2+1][0] == 1:
                    emb_labeled_1.append((key_emb, emb_pair_mean))
                elif pairs[j*2][0] == 0 and pairs[j*2+1][0] == 0:
                    emb_labeled_0.append((key_emb, emb_pair_mean))
        
        for key_emb, emb in emb_labeled_1 + emb_labeled_0:
            if key_emb not in stats_embeddings:
                stats_embeddings[key_emb] = {
                    'mean': np.mean(emb[:, i]),
                    'std': np.std(emb[:, i]),
                    'min': np.min(emb[:, i]),
                    'max': np.max(emb[:, i]),
                    'median': np.median(emb[:, i]),
                    'q1': np.percentile(emb[:, i], 25),
                    'q3': np.percentile(emb[:, i], 75)
                }
    return stats_embeddings

def cosine_similarity_numpy(emb1, emb2):
    return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def cosine_similarity_sklearn(emb1, emb2):
    return cosine_similarity(emb1, emb2)[0][0]

def mahalanobis_distance(vec1, vec2, cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return mahalanobis(vec1, vec2, inv_cov_matrix)

def calculate_mahalanobis_distances(embeddings, pairs):
    
    distances = []
    for pair in pairs:
        sample1_embeddings = embeddings[pair[3]]
        sample2_embeddings = embeddings[pair[4]]
        distance_per_model = []
        cov_matrix = np.cov(np.array(sample1_embeddings).T)
        for emb1, emb2 in zip(sample1_embeddings, sample2_embeddings):
            dist = mahalanobis_distance(emb1, emb2, cov_matrix)
            distance_per_model.append(dist)
        distances.append(distance_per_model)
    return distances

def cosine_similarity_scipy(emb1, emb2):
    return 1 - distance.cosine(emb1[0], emb2[0])

def calculate_cosine_distances(embeddings, pairs):

    distances = []
    for pair in pairs:
        
        sample1_embeddings = embeddings[pair[3]]
        sample2_embeddings = embeddings[pair[4]]
        distance_per_model = []

        for emb1, emb2 in zip(sample1_embeddings, sample2_embeddings):

            dis_scipy = cosine_similarity_scipy(emb1, emb2)           
            # distance = cosine_similarity_numpy(emb1, emb2)
            # dis_sklear = cosine_similarity_sklearn(emb1, emb2)
            
            distance_per_model.append(dis_scipy)

        distances.append(distance_per_model)

    return distances

def binary_classification(cosine_distances, pairs):
    
    ground_truth = [pair[0] for pair in pairs]

    accuracies = []
    conf_matrices = []
    reports = []

    for i in range(len(cosine_distances[0])):
        
        model_distances = [dist[i] for dist in cosine_distances]
        predictions = [1 if distance > 0.5 else 0 for distance in model_distances]

        accuracy = accuracy_score(ground_truth, predictions)
        conf_matrix = confusion_matrix(ground_truth, predictions)
        report = classification_report(ground_truth, predictions)

        accuracies.append(accuracy)
        conf_matrices.append(conf_matrix)
        reports.append(report)

    return accuracies, conf_matrices, reports

def evaluate_classification(cosine_distances, pairs, threshold):
    
    binary_ground_truth = [pair[0] for pair in pairs] 
    multiclass_ground_truth = [pair[1][-2:] for pair in pairs if pair[0] == 1] 
    
    accuracies_b = []
    conf_matrices_b = []
    reports_b = []
    eer_values = []
    bc_p = []
    
    accuracies_mc = []
    conf_matrices_mc = []
    reports_mc = []
    mc_p = []
    
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        
        bc_predictions = [1 if distance > threshold else 0 for _, distance in enumerate(model_distances)]

        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions)
        report_b = classification_report(binary_ground_truth, bc_predictions, output_dict=True)
        fpr, tpr, _ = roc_curve(binary_ground_truth, bc_predictions)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        accuracies_b.append(accuracy_b)
        conf_matrices_b.append(conf_matrix_b)
        reports_b.append(report_b)
        eer_values.append(eer)
        bc_p.append(bc_predictions)
        
        mc_pair = []
        p_i = []
        b_max_pred = []
        for i, dis in enumerate(model_distances):
            mc_pair.append(dis)
            if (i+1) % 5 == 0:
                b_max_pred.append([mc_pair, mc_pair.index(max(mc_pair))])
                p_i.append(model_distances.index(max(mc_pair)))
                mc_pair = []
        
        mlc_predictions = [int(pairs[index][2][-2:]) for index in p_i]
        mc_p.append(mlc_predictions)
    
        report = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
        accuracy = accuracy_score(multiclass_ground_truth, mlc_predictions)
        conf_matrix = confusion_matrix(multiclass_ground_truth, mlc_predictions)
        
        accuracies_mc.append(accuracy)
        conf_matrices_mc.append(conf_matrix)
        reports_mc.append(report)
        
    return mc_p, accuracies_mc, reports_mc, conf_matrices_mc, bc_p, accuracies_b, reports_b, conf_matrices_b, eer_values

def evaluate_classification_binary_closet(cosine_distances, pairs, threshold):
    
    binary_ground_truth = [pair[0] for pair in pairs]     
    balanced_accuracies_b = []
    accuracies_b = []
    conf_matrices_b = []
    reports_b = []
    eer_values = []
    
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        
        bc_predictions = [1 if distance > threshold[i_cos] else 0 for _, distance in enumerate(model_distances)]

        accuracy = accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix = confusion_matrix(binary_ground_truth, bc_predictions)
        report = classification_report(binary_ground_truth, bc_predictions, output_dict=True)
        balanced_accuracy = balanced_accuracy_score(binary_ground_truth, bc_predictions)
        
        fpr, tpr, _ = roc_curve(binary_ground_truth, bc_predictions)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        accuracies_b.append(accuracy)
        conf_matrices_b.append(conf_matrix)
        reports_b.append(report)
        eer_values.append(eer)
        balanced_accuracies_b.append(balanced_accuracy)
        
    return balanced_accuracies_b, accuracies_b, reports_b, conf_matrices_b, eer_values

def evaluate_classification_multiclass_closet_max(cosine_distances, pairs):
    
    multiclass_ground_truth = [pair[1][-2:] for pair in pairs if pair[0] == 1] 
    
    accuracies_mc = []
    balanc_accuracies_mc = []
    
    conf_matrices_mc = []
    reports_mc = []
    mc_p = []
    
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
        
        mlc_predictions = [str(pairs[index][2][-2:]) for index in p_i]
        mc_p.append(mlc_predictions)
    
        report = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
        accuracy = accuracy_score(multiclass_ground_truth, mlc_predictions)
        conf_matrix = confusion_matrix(multiclass_ground_truth, mlc_predictions)
        balanced_accuracy = balanced_accuracy_score(multiclass_ground_truth, mlc_predictions) 
        
        balanc_accuracies_mc.append(balanced_accuracy)
        accuracies_mc.append(accuracy)
        conf_matrices_mc.append(conf_matrix)
        reports_mc.append(report)
        
    return mc_p, balanc_accuracies_mc, accuracies_mc, reports_mc, conf_matrices_mc

def evaluate_classification_per_ss_B(cosine_distances, pairs, threshold, ss_n, k_way, q_no_q, q_c_n, q_c_size, binary_ground_truth):
    
    binary_ground_truth = [pair[0] for pair in pairs]     
    accuracies_b_list = []
    balanc_accuracies_b_list = []
    conf_matrices_b_list = []
    reports_b_list = []
    eer_list = []
    bc_p = []
    
    num_samples = len(cosine_distances)
    incd_all = []
 
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        if q_no_q:
            threshold_values = [threshold[i_cos][i] for i in range(ss_n*q_c_n) for _ in range(q_c_size*k_way)]
        else:
            threshold_values = [threshold[i_cos][i] for i in range(ss_n*q_c_size*q_c_n) for _ in range(k_way)]
        
        bc_predictions = [1 if distance > threshold_values[i_tresh] else 0 for i_tresh, distance in enumerate(model_distances)]

        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions)
        bala_accuracy_b = balanced_accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions)
        report_b = classification_report(binary_ground_truth, bc_predictions, output_dict=True)
        
        fpr, tpr, _ = roc_curve(binary_ground_truth, bc_predictions)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        eer_list.append(eer)
        accuracies_b_list.append(accuracy_b)
        balanc_accuracies_b_list.append(bala_accuracy_b)
        conf_matrices_b_list.append(conf_matrix_b)
        reports_b_list.append(report_b)
        bc_p.append(bc_predictions)
                
    return bc_p, accuracies_b_list, reports_b_list, conf_matrices_b_list, eer_list, balanc_accuracies_b_list

def evaluate_classification_per_ss(cosine_distances, pairs, threshold, ss_n, k_way, q_no_q, q_c_n, q_c_size,  support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels):
    
    binary_ground_truth = [pair[0] for pair in pairs]     
    accuracies_b_list = []
    balanc_accuracies_b_list = []
    conf_matrices_b_list = []
    reports_b_list = []
    eer_list = []
    bc_p = []
    
    num_samples = len(cosine_distances)
    incd_all = []
 
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        if q_no_q:
            threshold_values = [threshold[i_cos][i] for i in range(ss_n*q_c_n) for _ in range(q_c_size*k_way)]
        else:
            threshold_values = [threshold[i_cos][i] for i in range(ss_n*q_c_size*q_c_n) for _ in range(k_way)]
        
        bc_predictions = [1 if distance > threshold_values[i_tresh] else 0 for i_tresh, distance in enumerate(model_distances)]

        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions)
        bala_accuracy_b = balanced_accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions)
        report_b = classification_report(binary_ground_truth, bc_predictions, output_dict=True)
        
        fpr, tpr, _ = roc_curve(binary_ground_truth, bc_predictions)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        eer_list.append(eer)
        accuracies_b_list.append(accuracy_b)
        balanc_accuracies_b_list.append(bala_accuracy_b)
        conf_matrices_b_list.append(conf_matrix_b)
        reports_b_list.append(report_b)
        bc_p.append(bc_predictions)
        
        
        multiclass_predictions_list_acc = []
        # thresholdss_ = [duplicated_threshold[i_cos][num_i] for num_i in range(ss_n*q_c_n*q_c_size) for _ in range(k_way)]
        binary_predictions_list_acc = [1 if distance[i_cos] > threshold_values[c_i] else 0 for c_i, distance in enumerate(cosine_distances)]
        
        # Iterate over each quintet of samples
        for i in range(0, num_samples, k_way):
            quintet_distances = [dista[i_cos] for dista in cosine_distances[i:i+k_way]]
            quintet_thresholds_acc = threshold_values[i:i+k_way]
            
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_acc)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_acc = ['50']
        
            multiclass_predictions_list_acc.extend(quintet_predictions_acc)
        
        multiclass_predictions_from_start = []
        multiclass_predictions_from_start.append([id2label_map[int(cla)] for cla in multiclass_predictions_list_acc])
        multiclass_predictions_from_start = multiclass_predictions_from_start[0]
        
        accuracy_B = accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        bala_accuracy_B = balanced_accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        balanc_accuracy_m = balanced_accuracy_score(multi_ground_truth, multiclass_predictions_from_start)
        conf_matrix_m = confusion_matrix(multi_ground_truth, multiclass_predictions_from_start, labels=class_labels)
        report_m = classification_report(multi_ground_truth, multiclass_predictions_from_start, output_dict=True)
        
        incd_all.append([balanc_accuracy_m, bala_accuracy_B, accuracy_B, conf_matrix_m, report_m])
        
    return bc_p, accuracies_b_list, reports_b_list, conf_matrices_b_list, eer_list, balanc_accuracies_b_list, incd_all

def evaluate_classification_per_cat_ss(cosine_distances, cosine_distances_NO_Q,  pairs, ss_threshold, ss_n, 
                                       k_way, q_c_n, q_c_size,  support_set_classes, id2label_map, 
                                       binary_ground_truth, multi_ground_truth, class_labels):
    
    thresholds = [ss_threshold[i * k_way: (i + 1) * k_way] for i in range(k_way+1)]
    class_thresholds = {}
    for i, class_threshold in enumerate(thresholds):
        class_thresholds[f"class_{i + 1}_t"] = class_threshold
    
    num_samples = len(cosine_distances)
    incd_all = []
    
    for i_cos in range(len(cosine_distances[0])):
        
        threshold_values = []
        for c_i, threshold_c in enumerate(thresholds[i_cos]):
            threshold_values.append([threshold_c[i] for i in range(ss_n) for _ in range(q_c_n*q_c_size*k_way)])
        multiclass_predictions_list_acc = []
        
        # Iterate over each quintet of samples
        for i in range(0, num_samples, k_way):
            
            quintet_distances = [dista[i_cos] for dista in cosine_distances[i:i+k_way]]            
            category_threshold = []
            
            for j in range(k_way):
                category_threshold.append([threshold_values[j][i:i+k_way][0], j+1])
            
            max_tres = max(category_threshold)
            min_tres = min(category_threshold)
            
            cat_tres_norm = [d[0] / (max(category_threshold)[0] - min(category_threshold)[0]) for d in category_threshold]
            max_tres_n = max(cat_tres_norm)
            min_tres_n = min(cat_tres_norm)
            
            max_distance = max(quintet_distances)  # Find the maximum distance
            min_distance = min(quintet_distances)
            
            if min_tres[0] <= max_distance:
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_acc = ['50']
        
            multiclass_predictions_list_acc.extend(quintet_predictions_acc)
      
        multiclass_predictions_from_start = []
        multiclass_predictions_from_start.append([id2label_map[int(cla)] for cla in multiclass_predictions_list_acc])
        multiclass_predictions_from_start = multiclass_predictions_from_start[0]
    
        balanc_accuracy_m = balanced_accuracy_score(multi_ground_truth, multiclass_predictions_from_start)
        conf_matrix_m = confusion_matrix(multi_ground_truth, multiclass_predictions_from_start, labels=class_labels)
        report_m = classification_report(multi_ground_truth, multiclass_predictions_from_start, output_dict=True)
        
        incd_all.append([balanc_accuracy_m, conf_matrix_m, report_m])
        
    return incd_all

def classification_per_ss_mc(cosine_distances, pairs, threshold, ss_n, k_way, q_c_n, support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels):
    
    multiclass_predictions_all_list = []
    binary_predictions_all_list = []
    num_samples = len(cosine_distances)
    incd_all = []
    for i_cos in range(len(cosine_distances[0])):
        multiclass_predictions_list_acc = []
        thresholdss_ = [threshold[i_cos][num_i] for num_i in range(ss_n*q_c_n) for _ in range(k_way)]
        binary_predictions_list_acc = [1 if distance[i_cos] > thresholdss_[c_i] else 0 for c_i, distance in enumerate(cosine_distances)]
        
        # Iterate over each quintet of samples
        for i in range(0, num_samples, k_way):
            quintet_distances = [dista[i_cos] for dista in cosine_distances[i:i+k_way]]
            quintet_thresholds_acc = thresholdss_[i:i+k_way]
            
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_acc)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_acc = ['50']
        
            multiclass_predictions_list_acc.extend(quintet_predictions_acc)
        
        multiclass_predictions_from_start = []
        multiclass_predictions_from_start.append([id2label_map[int(cla)] for cla in multiclass_predictions_list_acc])
        multiclass_predictions_from_start = multiclass_predictions_from_start[0]
        
        accuracy_B = accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        bala_accuracy_B = balanced_accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        balanc_accuracy_m = balanced_accuracy_score(multi_ground_truth, multiclass_predictions_from_start)
        conf_matrix_m = confusion_matrix(multi_ground_truth, multiclass_predictions_from_start, labels=class_labels)
        report_m = classification_report(multi_ground_truth, multiclass_predictions_from_start, output_dict=True)
        
        incd_all.append([bala_accuracy_B, accuracy_B, conf_matrix_m, report_m, balanc_accuracy_m])
        binary_predictions_all_list.append(binary_predictions_list_acc)
        multiclass_predictions_all_list.append(multiclass_predictions_from_start)
        
    return multiclass_predictions_all_list, binary_predictions_all_list, incd_all

def classification_per_ss_CAT_mc(cosine_distances, pairs, threshold, ss_n, k_way, q_c_n, q_c_size, support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels):
    
    multiclass_predictions_all_list = []
    binary_predictions_all_list = []
    num_samples = len(cosine_distances)
    incd_all = []
    
    thresholds = [threshold[i * k_way: (i + 1) * k_way] for i in range(k_way+1)]
    num_samples = len(cosine_distances)
    incd_all = []
    
    for i_cos in range(len(cosine_distances[0])):
       
        multiclass_predictions_list_acc = []
        threshold_values = []
        for c_i, threshold_c in enumerate(thresholds[i_cos]):
            threshold_values.append([threshold_c[i] for i in range(ss_n) for _ in range(q_c_n*q_c_size*k_way)])
        multiclass_predictions_list_acc = []
        
        binary_predictions_list_acc = [1 if distance[i_cos] > max([va[c_i] for va in threshold_values]) else 0 for c_i, distance in enumerate(cosine_distances)]
        
        # Iterate over each quintet of samples
        for i in range(0, num_samples, k_way):
            
            quintet_distances = [dista[i_cos] for dista in cosine_distances[i:i+k_way]]            
            category_threshold = []
            
            for j in range(k_way):
                category_threshold.append([threshold_values[j][i:i+k_way][0], j+1])
            
            max_tres = max(category_threshold)
            min_tres = min(category_threshold)
            
            cat_tres_norm = [d[0] / (max(category_threshold)[0] - min(category_threshold)[0]) for d in category_threshold]
            max_tres_n = max(cat_tres_norm)
            min_tres_n = min(cat_tres_norm)
            
            max_distance = max(quintet_distances)  # Find the maximum distance
            min_distance = min(quintet_distances)
            
            if min_tres[0] <= max_distance:
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_acc = ['50']
        
            multiclass_predictions_list_acc.extend(quintet_predictions_acc)
            
        multiclass_predictions_from_start = []
        multiclass_predictions_from_start.append([id2label_map[int(cla)] for cla in multiclass_predictions_list_acc])
        multiclass_predictions_from_start = multiclass_predictions_from_start[0]
        
        accuracy_B = accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        bala_accuracy_B = balanced_accuracy_score(binary_ground_truth, binary_predictions_list_acc)
        balanc_accuracy_m = balanced_accuracy_score(multi_ground_truth, multiclass_predictions_from_start)
        conf_matrix_m = confusion_matrix(multi_ground_truth, multiclass_predictions_from_start, labels=class_labels)
        report_m = classification_report(multi_ground_truth, multiclass_predictions_from_start, output_dict=True)
        
        incd_all.append([bala_accuracy_B, accuracy_B, conf_matrix_m, report_m, balanc_accuracy_m])
        binary_predictions_all_list.append(binary_predictions_list_acc)
        multiclass_predictions_all_list.append(multiclass_predictions_from_start)
        
    return multiclass_predictions_all_list, binary_predictions_all_list, incd_all

def plot_confusion_matrix(conf_matrix, model_name, path):

    plt.figure(figsize=(15, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(path, f'confusion_matrix_{model_name}.png'), bbox_inches='tight')
    plt.show()

def plot_classification_report(report, model_name, path):
    
    lines = report.split('\n')
    classes = []
    plot_matrix = []

    for line in lines[2 : (len(lines) - 5)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plot_matrix.append(v)

    plt.figure(figsize=(15, 10))
    sns.heatmap(plot_matrix, annot=True, cmap='coolwarm',
                cbar=False, xticklabels=['precision', 'recall', 'f1-score'], yticklabels=classes)
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title(f'Classification Report for {model_name}')
    plt.savefig(os.path.join(path, f'classification_report_{model_name}.png'), bbox_inches='tight')
    plt.show()

def plot_combined(conf_matrix, report, model_name, path):
    
    # Prepare data for classification report
    lines = report.split('\n')
    classes = []
    plot_matrix = []

    for line in lines[2: (len(lines) - 5)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plot_matrix.append(v)

    # Create a grid for both plots
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Plot confusion matrix
    ax1 = plt.subplot(gs[0])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'Confusion Matrix for {model_name}')
    
    # Plot classification report
    ax2 = plt.subplot(gs[1])
    sns.heatmap(plot_matrix, annot=True, cmap='coolwarm', cbar=False, xticklabels=['precision', 'recall', 'f1-score'], yticklabels=classes, ax=ax2)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Classes')
    ax2.set_title(f'Classification Report for {model_name}')

    # Save and show the combined plot
    plt.savefig(os.path.join(path, f'combined_{model_name}.png'), bbox_inches='tight')
    plt.show()

def plot_scors(models_names, classifications_repo, acc, tresh, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    f1_scores = []
    recalls_scores = []
    precisions_scores = []
    
    in_list = ['weighted avg']
    for i, report in enumerate(classifications_repo):
        f1_scores_per_report = []
        recall_scores_per_report = []
        precision_scores_per_report = []
        
        for label in report.keys():
            if label in in_list:
                f1_scores_per_report.append(report[label]['f1-score'])
                recall_scores_per_report.append(report[label]['recall'])
                precision_scores_per_report.append(report[label]['precision'])
                
        recalls_scores.append(recall_scores_per_report[0])
        f1_scores.append(f1_scores_per_report[0])
        precisions_scores.append(precision_scores_per_report[0])

    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n = 2
    x = np.arange(0, len(models_names)*n, n)
    width = 0.15
    
    space = 0.0015
    rects1 = ax.bar(x - 3*(width + space), f1_scores, width, alpha=0.6, capsize=10, label='weighted F1-score')
    # Add text above bars
    for i, v in enumerate(f1_scores):
        ax.text(i*n - 3*(width + space), v+0.02, f"{v:.4f}", ha='center')

    # Plot bar chart of mean precision score
    rects3 = ax.bar(x - width - space, precisions_scores, width, alpha=0.6, capsize=10, label='weighted Precision')
    # Add text above bars
    for i, v in enumerate(precisions_scores):
        ax.text(i*n - (width + space), v+0.04, f"{v:.4f}", ha='center')

    # Plot bar chart of multi-class accuracy
    rects6 = ax.bar(x + width + space, recalls_scores, width, alpha=0.6, capsize=10, label='weighted recall')
    # Add text above bars
    for i, v in enumerate(recalls_scores):
        ax.text(i*n + (width + space), v+0.07, f"{v:.4f}", ha='center')
    
    # Plot bar chart of multi-class accuracy
    rects6 = ax.bar(x + 3*(width + space), acc, width, alpha=0.6, capsize=10, label='Multi-class balanc_ACC')
    # Add text above bars
    for i, (v, t) in enumerate(zip(acc, tresh)):
        ax.text(i*n + 3*(width + space), v+0.1, f"T:{t:.2f}\nACC:{v:.4f}", ha='center')
        
    # for i, v in enumerate(acc):
    #     ax.text(i*n + 3*(width + space), v+0.1, f"{v:.4f}", ha='center')
   
    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('ACC')
    ax.set_ylim([0, 1.1])
    
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()
    
def plot_b_scors(acc_best_list, eer_best_list, prec_best_list, thresholds, models_names, path, lab):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_names))
    width = 0.2
    
    # rects1 = ax.bar(x, acc_best_list, width, alpha=0.6, capsize=10, label=lab)
    # # Add text above bars
    # for i, (v, t) in enumerate(zip(acc_best_list, thresholds[1])):
    #     ax.text(i, v+0.05, f"T: {t:.2f}\nACC: {v*100:.3f}%", ha='center')
        
    rects2 = ax.bar(x, eer_best_list, width, alpha=0.6, capsize=10, label=lab)
    # Add text above bars
    for i, (v, t) in enumerate(zip(eer_best_list, thresholds)):
        ax.text(i, v+0.05, f"T: {t:.2f}\nB_ACC: {v*100:.3f}", ha='center')

    # rects3 = ax.bar(x, prec_best_list, width, alpha=0.6, capsize=10, label=lab)
    # # Add text above bars
    # for i, (v, t) in enumerate(zip(prec_best_list, thresholds[1])):
    #     # ax.text(i-3*width/2, v+0.02, f"{v:.2f}\n±{f1_scores_std[i]:.2f}", ha='center')
    #     ax.text(i, v+0.05, f"T: {t:.2f}\npreci: {v*100:.3f}", ha='center')

    # # Plot bar chart of mean recall score
    # rects4 = ax.bar(x - width/2 - 0.20, accuracies_b[1], width,
    #                 alpha=0.6, capsize=10, label=thresholds[1])
    # # Add text above bars
    # for i, v in enumerate(accuracies_b[1]):
    #     # ax.text(i-width/2, v+0.02, f"{v:.2f}\n±{recall_scores_std[i]:.2f}", ha='center')
    #     ax.text(i-width/2 - 0.20, v+0.02, f"{v*100:.0f}", ha='center')

    # # Plot bar chart of mean precision score
    # rects5 = ax.bar(x + width/2 - 0.15, accuracies_b[2], width,
    #                 alpha=0.6, capsize=10, label=thresholds[2])
    # # Add text above bars
    # for i, v in enumerate(accuracies_b[2]):
    #     # ax.text(i+width/2, v+0.02, f"{v:.2f}\n±{precisions_scores_std[i]:.2f}", ha='center')
    #     ax.text(i+width/2 - 0.15, v+0.02, f"{v*100:.0f}", ha='center')
        
    # # Plot bar chart of multi-class accuracy
    # rects6 = ax.bar(x + 3*width/2 - 0.1, accuracies_b[3], width, alpha=0.6, capsize=10,
    #                 label=thresholds[3])
    # # Add text above bars
    # for i, v in enumerate(accuracies_b[3]):
    #     ax.text(i+3*width/2 - 0.1, v+0.02, f"{v*100:.0f}", ha='center')

    # # Plot bar chart of binary accuracy
    # rects7 = ax.bar(x + 5*width/2 - 0.05, accuracies_b[4], width, alpha=0.6, capsize=10,
    #                 label=thresholds[4])
    # # Add text above bars
    # for i, v in enumerate(accuracies_b[4]):
    #     ax.text(i+ 5*width/2 - 0.05, v+0.02, f"{v*100:.0f}", ha='center')



    # Add labels, title and axis ticks
    ax.set_title(f'best {lab} and coresponding treshold')
    ax.set_xlabel('Model')
    ax.set_ylabel(f'Binary {lab}')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='best')
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()
    
def plot_ss_scors(accuracies_b, accuracies_b_0, accuracies_b_per_ss_, sig_alfa, models_names, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.4
    space_between_bars = 0.55
    fs = 6
    x = np.arange(0, len(models_names)*fs, fs)

    rects1 = ax.bar(x - 2*width - space_between_bars, accuracies_b, width, alpha=0.6, capsize=10, label='Mean')
    # Add text above bars
    for i, (v, t) in enumerate(zip(accuracies_b, sig_alfa[0])):
        ax.text(i*fs - 2*width - space_between_bars, v+0.08, f"S:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)

    # rects2 = ax.bar(x - 2*width - space_between_bars, accuracies_b[1], width, alpha=0.6, capsize=10, label='5_threshol_mad')
    # # Add text above bars
    # for i, (v, t) in enumerate(zip(accuracies_b[1], sig_alfa[1])):
    #     ax.text(i*fs - 2*width - space_between_bars, v+0.25, f"A:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)

    rects4 = ax.bar(x , accuracies_b_0, width, alpha=0.6, capsize=10, label='Max')
    # Add text above bars
    for i, (v, t) in enumerate(zip(accuracies_b_0, sig_alfa[1])):
        ax.text(i*fs , v+0.02, f"A:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)
    
    # rects3 = ax.bar(x - width, accuracies_b_0[0], width, alpha=0.6, capsize=10, label='4_threshol_mad')
    # # Add text above bars
    # for i, (v, t) in enumerate(zip(accuracies_b_0[0], sig_alfa[2])):
    #     ax.text(i*fs - width, v+0.15, f"S:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)
    
    rects5 = ax.bar(x + 2*width + space_between_bars, accuracies_b_per_ss_, width, alpha=0.6, capsize=10, label='MAD')
    # Add text above bars
    for i, (v, t) in enumerate(zip(accuracies_b_per_ss_, sig_alfa[2])):
        ax.text(i*fs + 2*width + space_between_bars, v+0.1, f"S:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)

    # rects6 = ax.bar(x + 3*width + 2*space_between_bars, accuracies_b_per_ss_[1], width, alpha=0.6, capsize=10, label='10_threshol_mad')
    # # Add text above bars
    # for i, (v, t) in enumerate(zip(accuracies_b_per_ss_[1], sig_alfa[5])):
    #     ax.text(i*fs + 3*width + 2*space_between_bars, v+0.13, f"S:{t:.2f}\n{v:.3f}", ha='center', fontsize=fs)

    # Add labels, title and axis ticks
    ax.set_xlim(-4, len(models_names)*fs - 0.5)
    ax.set_xlabel('Model')
    ax.set_ylabel('Multi-class balanced Acc')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')


    plt.show()
    
def plot_sieamis_b_scors(models_names, accuracies_b, path):
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_names))
    width = 0.2
    
    rects1 = ax.bar(x, accuracies_b, width,
                    alpha=0.6, capsize=10, label='binary_acc')
    
    # Add text above bars
    for i, v in enumerate(accuracies_b):
        ax.text(i, v+0.02, f"{v*100:.0f}", ha='center')
    
    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('Binary Accuracy')
    ax.set_ylim(0, 1.05)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()

def evaluate_sieamis_classification(predictions, binary_ground_truth, real_class):
    
    predictions_list = [sublist[0] for sublist in predictions]
    
    sigmo = torch.nn.Sigmoid()
    normalized_predictions_list = sigmo(torch.tensor(predictions_list)).tolist()
    
    classified_max_ssq_list = []
    for i in range(0, len(normalized_predictions_list), 5):
        group = normalized_predictions_list[i:i+5]
        max_value = max(group)
        group_classification = [1 if value == max_value else 0 for value in group]
        classified_max_ssq_list.extend(group_classification)
            
    accuracy_b = accuracy_score(binary_ground_truth, classified_max_ssq_list)
    balanc_accuracy_b = balanced_accuracy_score(binary_ground_truth, classified_max_ssq_list)
    conf_matrix_b = confusion_matrix(binary_ground_truth, classified_max_ssq_list)
    report_b = classification_report(binary_ground_truth, classified_max_ssq_list, output_dict=True)
    binary = [balanc_accuracy_b, accuracy_b, conf_matrix_b, report_b]
    
    multiclass_ground_truth = []
    for _, ss in enumerate(real_class):
        for i_ss_fs, ss_fs in enumerate(ss[0]):
            if i_ss_fs % 5 == 0:
                multiclass_ground_truth.append(ss_fs[-2:])
        
    mlc_predictions = []
    real_class_lables = [rcl[-2:] for rc in real_class for rcl in rc[1]]
    for i in range(0, len(normalized_predictions_list), 5):
        
        group_predictions = normalized_predictions_list[i:i+5]
        group_labels = real_class_lables[i:i+5]
        
        group_max_index = max(range(len(group_predictions)), key=group_predictions.__getitem__)
        predicted_label = group_labels[group_max_index]
        mlc_predictions.append(predicted_label)
        
    accuracy_multi = accuracy_score(multiclass_ground_truth, mlc_predictions)
    balanc_accuracy_m = balanced_accuracy_score(multiclass_ground_truth, mlc_predictions)
    conf_matrix_multi = confusion_matrix(multiclass_ground_truth, mlc_predictions)
    report_multi = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
    
    multi = [balanc_accuracy_m, accuracy_multi, conf_matrix_multi, report_multi]

    return binary, multi

def plot_sieamis_scors(models_names, classifications_repo, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    f1_scores = []
    f1_scores_mean = []
    f1_scores_std = []
    labels = []
    
    for i, report in enumerate(classifications_repo):
        f1_scores_per_report = []
        for label in report[1][3].keys():
            if label.isdigit():
                f1_scores_per_report.append(report[1][3][label]['f1-score'])
                if i == 0:
                    labels.append(label)
        f1_scores.append(f1_scores_per_report)
        f1_scores_mean.append(np.mean(f1_scores_per_report))
        f1_scores_std.append(np.std(f1_scores_per_report))
    
    n_classes = len(labels)
    
    recall_scores_mean = []
    recall_scores_std = []
    recalls_scores = []
    balanced_accuracy = []
    for report in classifications_repo:
        recall_scores_per_report = []
        for label in report[1][3].keys():
            if label.isdigit():
                recall_scores_per_report.append(report[1][3][label]['recall'])
        recalls_scores.append(recall_scores_per_report)
        recall_scores_mean.append(np.mean(recall_scores_per_report))
        recall_scores_std.append(np.std(recall_scores_per_report))
        balanced_accuracy.append((1/n_classes) * sum(recall_scores_per_report))
        
    precisions_scores_mean = []
    precisions_scores_std = []
    precisions_scores = []
    for report in classifications_repo:
        precision_scores_per_report = []
        for label in report[1][3].keys():
            if label.isdigit():
                precision_scores_per_report.append(report[1][3][label]['precision'])
        precisions_scores.append(precision_scores_per_report)
        precisions_scores_mean.append(np.mean(precision_scores_per_report))
        precisions_scores_std.append(np.std(precision_scores_per_report))

    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_names))
    width = 0.1

    rects1 = ax.bar(x - 3*width/2 - 0.25, f1_scores_mean, width,
                    alpha=0.6, capsize=10, label='Mean F1-score')
    # Add text above bars
    for i, v in enumerate(f1_scores_mean):
        # ax.text(i-3*width/2, v+0.02, f"{v:.2f}\n±{f1_scores_std[i]:.2f}", ha='center')
        ax.text(i-3*width/2 - 0.25, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of mean recall score
    rects2 = ax.bar(x - width/2 - 0.20, recall_scores_mean, width,
                    alpha=0.6, capsize=10, label='Mean Recall')
    # Add text above bars
    for i, v in enumerate(recall_scores_mean):
        # ax.text(i-width/2, v+0.02, f"{v:.2f}\n±{recall_scores_std[i]:.2f}", ha='center')
        ax.text(i-width/2 - 0.20, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of mean precision score
    rects3 = ax.bar(x + width/2 - 0.15, precisions_scores_mean, width,
                    alpha=0.6, capsize=10, label='Mean Precision')
    # Add text above bars
    for i, v in enumerate(precisions_scores_mean):
        # ax.text(i+width/2, v+0.02, f"{v:.2f}\n±{precisions_scores_std[i]:.2f}", ha='center')
        ax.text(i+width/2 - 0.15, v+0.02, f"{v*100:.0f}", ha='center')
        
    # Plot bar chart of multi-class accuracy
    rects6 = ax.bar(x + 3*width/2 - 0.1, balanced_accuracy, width, alpha=0.6, capsize=10,
                    label='Multi-class balanced_acc')
    # Add text above bars
    for i, v in enumerate(balanced_accuracy):
        ax.text(i+3*width/2 - 0.1, v+0.02, f"{v*100:.0f}", ha='center')


    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend()
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()

def evaluate_classification_openset(multiclass_predictions, binary_predictions, multi_ground_truth, binary_ground_truth):

    accuracies_b = []
    conf_matrices_b = []
    reports_b = []
    binary_bala_acc = []

    accuracies_m = []
    conf_matrices_m = []
    reports_m = []
    mc_bala_acc = []
    # top1_accuracy_list = []
    
    for i_cos in range(6):

        # Binary classification
        # bc_predictions_err = binary_predictions[i_cos][0]
        # bc_predictions_preci = binary_predictions[i_cos][1]
        # bc_predictions_acc = binary_predictions[i_cos][2]
        bc_predictions_balanc_acc = binary_predictions[i_cos][3]
        
        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions_balanc_acc)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions_balanc_acc)
        report_b = classification_report(binary_ground_truth, bc_predictions_balanc_acc, output_dict=True)
        b_a_bi = balanced_accuracy_score(binary_ground_truth, bc_predictions_balanc_acc)

        accuracies_b.append(accuracy_b)
        conf_matrices_b.append(conf_matrix_b)
        reports_b.append(report_b)
        binary_bala_acc.append(b_a_bi)

        # Multiclass classification
        mc_predictions = multiclass_predictions[i_cos]

        # Extract unique class labels from multi_ground_truth
        class_labels = list(set(multi_ground_truth))
        class_labels.remove('unknown')
        class_labels.append('unknown')

        accuracy_m = accuracy_score(multi_ground_truth, mc_predictions)
        conf_matrix_m = confusion_matrix(multi_ground_truth, mc_predictions, labels=class_labels)
        report_m = classification_report(multi_ground_truth, mc_predictions, output_dict=True)
        b_a_mc = balanced_accuracy_score(multi_ground_truth, mc_predictions)
        
        accuracies_m.append(accuracy_m)
        conf_matrices_m.append(conf_matrix_m)
        reports_m.append(report_m)
        mc_bala_acc.append(b_a_mc)
        
    return accuracies_b, reports_b, conf_matrices_b, binary_bala_acc, accuracies_m, reports_m, conf_matrices_m, mc_bala_acc, class_labels

def calculate_fix_threshold_openset(distances, labels, num_thresholds=100):
    
    thresholds = np.linspace(min(distances), max(distances), num=num_thresholds)
    all_best_thresholds = []
    
    for i_cos in range(len(distances[0])):
        accuracies = []
        precisions = []
        eer_values = []
        balanced_accuracy_score_values = []
        
        for threshold in thresholds:
            predictions = [1 if distance[i_cos] > threshold[i_cos] else 0 for distance in distances]
            accuracy = accuracy_score(labels, predictions)
            precision = classification_report(labels, predictions, output_dict=True)['weighted avg']['precision']
            balanced_acc = balanced_accuracy_score(labels, predictions)
            fpr, tpr, _ = roc_curve(labels, predictions)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
            accuracies.append(accuracy)
            precisions.append(precision)
            eer_values.append(eer)
            balanced_accuracy_score_values.append(balanced_acc)
        
        best_accuracy_index = np.argmax(accuracies)
        best_precision_index = np.argmax(precisions)
        best_balanced_acc_index = np.argmax(balanced_accuracy_score_values)
        best_eer_index = np.argmin(eer_values)
        
        best_accuracy_threshold = thresholds[best_accuracy_index][i_cos]
        best_precision_threshold = thresholds[best_precision_index][i_cos]
        best_balanced_acc_threshold = thresholds[best_balanced_acc_index][i_cos]
        best_eer_threshold = thresholds[best_eer_index][i_cos]
        
        all_best_thresholds.append([best_balanced_acc_threshold, best_accuracy_threshold, best_precision_threshold, best_eer_threshold])
    
    return all_best_thresholds

def calculate_pers_threshold_openset(distances, distances_no_q, pairs, ss_param, models_names, support_set_num, k_way, q_c_n, q_c_size,  support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels, num_thresholds=1000):
    
    ss_param_no_q, ss_param_q = ss_param
    pairs_q, pairs_no_q = pairs
    
    tresh_sig_const = np.linspace(0, 6, num=num_thresholds)
    tresh_alfa_const = np.linspace(0, 6, num=num_thresholds)
    
    acc_list_5_mss, acc_list_5_mad, acc_list_4_mss, acc_list_4_mad, acc_list_10_mss, acc_list_10_mass_, err_list_5_mss, err_list_5_mad, err_list_4_mss, err_list_4_mad, err_list_10_mss, err_list_10_mass_ = [[] for _ in range(12)]
    acc_list_5_mss__ = []   
    acc_list_5_mad__= []
    acc_list_4_mss__= []
    acc_list_4_mad__= []
    acc_list_10_mss__= []
    acc_list_10_mass___= []
    all_tresh_ = []
    for index_const, (sig, alf) in enumerate(zip(tresh_sig_const, tresh_alfa_const)):
        ss_tresholds_all, ss_tresholds_all_0, ss_tresholds_no_q_max_sig_std, ss_tresholds_no_q_mean_sig_std, ss_tresholds_all_max_alfa_diff, ss_tresholds_all_0_max_alfa_diff = [[] for _ in range(6)]
        for i in range(len(models_names)):
            
            ss_tresholds_all.append([mad + sig*std for mad, std in zip(ss_param_q['MAD'][i], ss_param_q['f_s_dif'][i])])                    
            ss_tresholds_all_max_alfa_diff.append([max_ - sig*diff_ for max_, diff_ in zip(ss_param_q['max'][i], ss_param_q['f_s_dif'][i])])

            ss_tresholds_all_0.append([mean + sig*std for mean, std in zip(ss_param_q['MAD_0'][i], ss_param_q['f_s_dif'][i])])
            ss_tresholds_all_0_max_alfa_diff.append([max_ + sig*diff_ for max_, diff_ in zip(ss_param_q['max0'][i], ss_param_q['f_s_dif'][i])])
            
            ss_tresholds_no_q_mean_sig_std.append([value for value in [mean[0] + sig*std[0] for mean, std in zip(ss_param_no_q['MAD'][i], ss_param_no_q['std_all'][i]) for _ in range(q_c_n[1]*q_c_size[1])]])
            ss_tresholds_no_q_max_sig_std.append([max_[0] + sig*std for max_, std in zip([value for val in ss_param_no_q['max'][i] for value in [val] * (q_c_n[1]* q_c_size[1])], ss_param_q['f_s_dif'][i])])
            
        bc_p_mss, accuracies_b_mss, reports_b_mss, conf_matrices_b_mss, eer_mss, bala_mss, incd_all_mc_mss = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_all, support_set_num[0], k_way[0], True, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        bc_p_MAD, accuracies_b_mad, reports_b_mad, _, eer_mad, bala_mad, incd_all_mc_mad = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_all_max_alfa_diff, support_set_num[0], k_way[0], True, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        
        bc_p_0_mss, accuracies_b_0_mss, reports_b_0_mss, _, eer_0_mss, bala_0_mss, incd_all_mc_0_mss = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_all_0, support_set_num[0], k_way[0], True, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        bc_p_0_MAD, accuracies_b_0_mad, reports_b_0_mad, _, eer_0_mad, bala_0_mad, incd_all_mc_0_mad = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_all_0_max_alfa_diff, support_set_num[0], k_way[0], True, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        
        bc_p_ss, accuracies_b_per_ss, reports_b_ss, _, eer_ss, bala_ss, incd_all_mc_10_ss = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_no_q_mean_sig_std, support_set_num[0], k_way[0], False, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        bc_p_0_ss, accuracies_b_per_ss_max, reports_b_ss_max_, _, eer_ss_, bala_ss_max, incd_all_mc_10_max = evaluate_classification_per_ss(distances, pairs_q, ss_tresholds_no_q_max_sig_std, support_set_num[0], k_way[0], True, q_c_n[1], q_c_size[1], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        
        acc_list_5_mss.append(bala_mss)
        acc_list_5_mad.append(bala_mad)
        acc_list_4_mss.append(bala_0_mss)
        acc_list_4_mad.append(bala_0_mad)
        acc_list_10_mss.append(bala_ss)
        acc_list_10_mass_.append(bala_ss_max)
        
        acc_list_5_mss__.append([elem[0] for elem in incd_all_mc_mss])
        acc_list_5_mad__.append([elem[0] for elem in incd_all_mc_mad])
        acc_list_4_mss__.append([elem[0] for elem in incd_all_mc_0_mss])
        acc_list_4_mad__.append([elem[0] for elem in incd_all_mc_0_mad])
        acc_list_10_mss__.append([elem[0] for elem in incd_all_mc_10_ss])
        acc_list_10_mass___.append([elem[0] for elem in incd_all_mc_10_max])
        
        err_list_5_mss.append(eer_mss)
        err_list_5_mad.append(eer_mad)
        err_list_4_mss.append(eer_0_mss)
        err_list_4_mad.append(eer_0_mad)
        err_list_10_mss.append(eer_ss)
        err_list_10_mass_.append(eer_ss_)
        
        all_tresh_.append([sig, alf])
    
    best_indices_acc_mss = [[np.argmax([inner_list[i] for inner_list in acc_list_5_mss]), np.max([inner_list[i] for inner_list in acc_list_5_mss])] for i in range(len(acc_list_5_mss__[0]))]
    best_indices_acc_mad = [[np.argmax([inner_list[i] for inner_list in acc_list_5_mad]), np.max([inner_list[i] for inner_list in acc_list_5_mad])] for i in range(len(acc_list_5_mad__[0]))]
    
    best_indices_acc_0_mss = [[np.argmax([inner_list[i] for inner_list in acc_list_4_mss]), np.max([inner_list[i] for inner_list in acc_list_4_mss])] for i in range(len(acc_list_4_mss__[0]))]
    best_indices_acc_0_mad = [[np.argmax([inner_list[i] for inner_list in acc_list_4_mad]), np.max([inner_list[i] for inner_list in acc_list_4_mad])] for i in range(len(acc_list_4_mad__[0]))]
    
    best_indices_acc_per_ss = [[np.argmax([inner_list[i] for inner_list in acc_list_10_mss]), np.max([inner_list[i] for inner_list in acc_list_10_mss])] for i in range(len(acc_list_10_mss__[0]))]
    best_indices_acc_ss_max = [[np.argmax([inner_list[i] for inner_list in acc_list_10_mass_]), np.max([inner_list[i] for inner_list in acc_list_10_mass_])] for i in range(len(acc_list_10_mass___[0]))]
    
    sig_5 = [all_tresh_[best_indices_acc_mss[i][0]][0] for i in range(len(best_indices_acc_mss))]
    alfa_5 = [all_tresh_[best_indices_acc_mad[i][0]][1] for i in range(len(best_indices_acc_mad))]
    
    sig_4 = [all_tresh_[best_indices_acc_0_mss[i][0]][0] for i in range(len(best_indices_acc_0_mss))]
    alfa_4 = [all_tresh_[best_indices_acc_0_mad[i][0]][1] for i in range(len(best_indices_acc_0_mad))]
    
    sig_10 = [all_tresh_[best_indices_acc_per_ss[i][0]][0] for i in range(len(best_indices_acc_per_ss))]
    sig_10_ = [all_tresh_[best_indices_acc_ss_max[i][0]][0] for i in range(len(best_indices_acc_ss_max))]
    
    sig_alfa_acc = [sig_5, alfa_5, sig_4, alfa_4, sig_10, sig_10_]
    
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
    
    return [sig_alfa_acc, sig_alfa_eer]

def calculate_pers_CAT_threshold_openset(distances, distances_no_q, pairs, ss_param, models_names, support_set_num, k_way, q_c_n, q_c_size,
                                         support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels, num_thresholds=1000):
    
    ss_param_no_q, ss_param_q = ss_param
    pairs_q, pairs_no_q = pairs
    
    tresh_sig_const = np.linspace(0, 6, num=num_thresholds)
    
    acc_list_5_MAX = []   
    acc_list_5_Mean = []
    acc_list_5_Median = []
    conf_acc_list_5_MAX = []   
    
    all_tresh_ = []
    
    for _, sig in enumerate(tresh_sig_const):

        ss_tresholds_no_q_mean = []
        ss_tresholds_no_q_max = []
        ss_tresholds_no_q_median = []
        
        for i in range(len(models_names)):
            for cat_ind in range(1, k_way[0]+1):
                ss_tresholds_no_q_mean.append([mean[cat_ind] + sig*std[cat_ind] for _, (mean, std) in enumerate(zip(ss_param_no_q['mean_all'][i], ss_param_no_q['std_all'][i]))])
                ss_tresholds_no_q_max.append([mean[cat_ind] + sig*std[cat_ind] for _, (mean, std) in enumerate(zip(ss_param_no_q['max'][i], ss_param_no_q['std_all'][i]))])
                ss_tresholds_no_q_median.append([mean[cat_ind] + sig*std[cat_ind] for _, (mean, std) in enumerate(zip(ss_param_no_q['MAD'][i], ss_param_no_q['std_all'][i]))])

        incd_all_mc_mean = evaluate_classification_per_cat_ss(distances, distances_no_q,  pairs_q, ss_tresholds_no_q_mean, support_set_num[0], k_way[0], q_c_n[1], q_c_size[0], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)                
        incd_all_mc_max = evaluate_classification_per_cat_ss(distances, distances_no_q, pairs_q, ss_tresholds_no_q_max, support_set_num[0], k_way[0], q_c_n[1], q_c_size[0], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
        incd_all_mc_median = evaluate_classification_per_cat_ss(distances, distances_no_q, pairs_q, ss_tresholds_no_q_median, support_set_num[0], k_way[0], q_c_n[1], q_c_size[0], support_set_classes, id2label_map, binary_ground_truth, multi_ground_truth, class_labels)
    
        acc_list_5_MAX.append([elem[0] for elem in incd_all_mc_max])
        acc_list_5_Mean.append([elem[0] for elem in incd_all_mc_mean])
        acc_list_5_Median.append([elem[0] for elem in incd_all_mc_median])
        conf_acc_list_5_MAX.append(incd_all_mc_max)
        all_tresh_.append(sig)
    
    best_indices_acc_max = [[np.argmax([inner_list[i] for inner_list in acc_list_5_MAX]), np.max([inner_list[i] for inner_list in acc_list_5_MAX])] for i in range(len(acc_list_5_MAX[0]))]
    best_indices_acc_Mean = [[np.argmax([inner_list[i] for inner_list in acc_list_5_Mean]), np.max([inner_list[i] for inner_list in acc_list_5_Mean])] for i in range(len(acc_list_5_Mean[0]))]
    best_indices_acc_Median = [[np.argmax([inner_list[i] for inner_list in acc_list_5_Median]), np.max([inner_list[i] for inner_list in acc_list_5_Median])] for i in range(len(acc_list_5_Median[0]))]
    
    sig_max = [all_tresh_[best_indices_acc_max[i][0]] for i in range(len(best_indices_acc_max))]
    sig_mean = [all_tresh_[best_indices_acc_max[i][0]] for i in range(len(best_indices_acc_max))]
    sig_median = [all_tresh_[best_indices_acc_max[i][0]] for i in range(len(best_indices_acc_max))]
    conf_MAX = [conf_acc_list_5_MAX[best_indices_acc_max[i][0]][i] for i in range(len(best_indices_acc_max))]
    
    return sig_max, sig_mean, sig_median, conf_MAX

def multiclass_binary_openset_classification(distances_real, thresholds, support_set_classes):
    
    multiclass_predictions_all_list = []
    binary_predictions_all_list = []
    num_samples = len(distances_real)
    
    for i_cos in range(len(distances_real[0])):
        multiclass_predictions_list_acc = []
        multiclass_predictions_list_balanc_acc = []
        multiclass_predictions_list_preci = []
        multiclass_predictions_list_err = []
        
        
        thresholdss_balanc_acc = [thresholds[i_cos][0]] * num_samples
        thresholdss_acc = [thresholds[i_cos][1]] * num_samples
        thresholdss_preci = [thresholds[i_cos][2]] * num_samples
        thresholdss_err = [thresholds[i_cos][3]] * num_samples
        
        binary_predictions_list_balanc_acc = [1 if distance[i_cos] > thresholds[i_cos][0] else 0 for distance in distances_real]
        binary_predictions_list_acc = [1 if distance[i_cos] > thresholds[i_cos][1] else 0 for distance in distances_real]
        binary_predictions_list_preci = [1 if distance[i_cos] > thresholds[i_cos][2] else 0 for distance in distances_real]
        binary_predictions_list_err = [1 if distance[i_cos] > thresholds[i_cos][3] else 0 for distance in distances_real]
        
        # Iterate over each quintet of samples
        for i in range(0, num_samples, 5):
            quintet_distances = [dista[i_cos] for dista in distances_real[i:i+5]]
            
            quintet_thresholds_eer = thresholdss_err[i:i+5]
            quintet_thresholds_preci = thresholdss_preci[i:i+5]
            quintet_thresholds_acc = thresholdss_acc[i:i+5]
            quintet_thresholds_balanc_acc = thresholdss_balanc_acc[i:i+5]
            
            
            # Check if any distance in the quintet exceeds the threshold
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_eer)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_eer = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_eer = ['50']
                
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_preci)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_preci = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_preci = ['50']
                
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_acc)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_acc = ['50']
            
            if any(distance > threshold for distance, threshold in zip(quintet_distances, quintet_thresholds_balanc_acc)):
                closest_class_index = np.argmax(quintet_distances)
                quintet_predictions_balanc_acc = [support_set_classes[i+closest_class_index]]
            else:
                quintet_predictions_balanc_acc = ['50']
            
            multiclass_predictions_list_err.append(quintet_predictions_eer)
            multiclass_predictions_list_preci.append(quintet_predictions_preci)
            multiclass_predictions_list_acc.append(quintet_predictions_acc)
            multiclass_predictions_list_balanc_acc.append(quintet_predictions_balanc_acc)
            
        binary_predictions_all_list.append([binary_predictions_list_err, binary_predictions_list_preci, binary_predictions_list_acc, binary_predictions_list_balanc_acc])
        multiclass_predictions_all_list.append([multiclass_predictions_list_err, multiclass_predictions_list_preci, multiclass_predictions_list_acc, multiclass_predictions_list_balanc_acc])
    
    return multiclass_predictions_all_list, binary_predictions_all_list

def plot_confusion_matrices(confusion_matrices, path, class_labels, single_plot=False):
    num_models = len(confusion_matrices)
    num_rows = 3
    num_cols = 2
    
    if single_plot:
        num_rows = num_models // num_cols
        num_cols = num_models // num_rows
        if num_models % num_rows != 0:
            num_rows += 1
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))    
    for i, ax in enumerate(axes.flatten()):
        if i < num_models:
            # Convert the confusion matrix to a DataFrame with class labels
            confusion_df = pd.DataFrame(confusion_matrices[i], index=class_labels, columns=class_labels)

            sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix {i+1}")
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
        else:
            ax.axis('off')
    
    fig.tight_layout()
    
    if single_plot:
        plt.savefig(f"{path}/combined_confusion_matrix.png", bbox_inches='tight')
        plt.show()
    else:
        for i, matrix in enumerate(confusion_matrices):
            confusion_df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix {i+1}")
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(f"{path}/confusion_matrix_{i+1}.png", bbox_inches='tight')
            plt.close()
