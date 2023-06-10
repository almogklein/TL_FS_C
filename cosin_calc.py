# from collections import Counter
# from typing import List, Tuple
# import csv
# import random

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #, average_precision_score, pairwise_distances  


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

def cosine_similarity_scipy(emb1, emb2):
    return 1 - distance.cosine(emb1[0], emb2[0])

def cosine_similarity_sklearn(emb1, emb2):
    return cosine_similarity(emb1, emb2)[0][0]

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

def evaluate_binary_classification(cosine_distances, pairs):
    
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
    bc_p = []
    
    accuracies_mc = []
    conf_matrices_mc = []
    reports_mc = []
    mAP_l = []
    mc_p = []
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        
        # max_value = max(model_distances)
        # min_value = min(model_distances)
        # model_distances = [(x - min_value) / (max_value - min_value) for x in model_distances]
        
        bc_predictions = [1 if distance > threshold else 0 for _, distance in enumerate(model_distances)]

        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions)
        report_b = classification_report(binary_ground_truth, bc_predictions, output_dict=True)

        accuracies_b.append(accuracy_b)
        conf_matrices_b.append(conf_matrix_b)
        reports_b.append(report_b)
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
        
        # # loop through the input list and append the indexes to the appropriate class list
        # class_indexes_truth = {k: [] for k in class_lookup.keys()}
        # class_pred = {k: [] for k in class_lookup.keys()}
        
        # for i, c in enumerate(multiclass_ground_truth):
        #     class_indexes_truth[str(c)].append(i)
        
        # for clas in class_indexes_truth.keys():
        #     class_pred[clas].append([mlc_predictions[i] for i in class_indexes_truth[clas]])
        
        # m_av_p = []
        # for clas in class_pred.keys():
        #     for value in class_pred[clas]:
                
        #     m_avp =  average_precision_score(class_indexes_truth[clas], class_pred[clas], average=None)
        #     m_av_p.append(m_avp)
        
        # mAP = sum(m_av_p)/len(m_av_p) #np.mean(m_av_p)
        
        report = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
        accuracy = accuracy_score(multiclass_ground_truth, mlc_predictions)
        conf_matrix = confusion_matrix(multiclass_ground_truth, mlc_predictions)
        
        accuracies_mc.append(accuracy)
        conf_matrices_mc.append(conf_matrix)
        reports_mc.append(report)
        # mAP_l.append(mAP)
        
    return mc_p, accuracies_mc, reports_mc, conf_matrices_mc, mAP_l, bc_p, accuracies_b, reports_b, conf_matrices_b

def evaluate_classification_per_ss(cosine_distances, pairs, threshold):
    

    binary_ground_truth = [pair[0] for pair in pairs] 
    
    multi_ground_truth = [pair[1][-2:] for pair in pairs if pair[0] == 1] 
    
    accuracies_b = []
    conf_matrices_b = []
    reports_b = []
    bc_p = []
 
    for i_cos in range(len(cosine_distances[0])):
        
        model_distances = [dist[i_cos] for dist in cosine_distances]
        threshold_values = [threshold[i_cos][i] for i in range(15000) for _ in range(5)]
        
        bc_predictions = [1 if distance > threshold_values[i_tresh] else 0 for i_tresh, distance in enumerate(model_distances)]

        accuracy_b = accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix_b = confusion_matrix(binary_ground_truth, bc_predictions)
        report_b = classification_report(binary_ground_truth, bc_predictions, output_dict=True)

        accuracies_b.append(accuracy_b)
        conf_matrices_b.append(conf_matrix_b)
        reports_b.append(report_b)
        bc_p.append(bc_predictions)
        
    return bc_p, accuracies_b, reports_b, conf_matrices_b

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

def plot_scors(models_names, classifications_repo, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    f1_scores = []
    f1_scores_mean = []
    f1_scores_std = []
    labels = []
    
    for i, report in enumerate(classifications_repo):
        f1_scores_per_report = []
        for label in report.keys():
            if label.isdigit():
                f1_scores_per_report.append(report[label]['f1-score'])
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
        for label in report.keys():
            if label.isdigit():
                recall_scores_per_report.append(report[label]['recall'])
        recalls_scores.append(recall_scores_per_report)
        recall_scores_mean.append(np.mean(recall_scores_per_report))
        recall_scores_std.append(np.std(recall_scores_per_report))
        balanced_accuracy.append((1/n_classes) * sum(recall_scores_per_report))
        
    precisions_scores_mean = []
    precisions_scores_std = []
    precisions_scores = []
    for report in classifications_repo:
        precision_scores_per_report = []
        for label in report.keys():
            if label.isdigit():
                precision_scores_per_report.append(report[label]['precision'])
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
    
def plot_b_scors(accuracies_b, person_ss_tres_, person_ss_tres, thresholds, models_names, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_names))
    width = 0.07
    
    rects1 = ax.bar(x - 7*width/2 - 0.35, person_ss_tres_, width,
                    alpha=0.6, capsize=10, label='personal_ss_5_threshold')
    # Add text above bars
    for i, v in enumerate(person_ss_tres_):
        # ax.text(i-3*width/2, v+0.02, f"{v:.2f}\n±{f1_scores_std[i]:.2f}", ha='center')
        ax.text(i-5*width/2 - 0.42, v+0.02, f"{v*100:.0f}", ha='center')
        
    rects1 = ax.bar(x - 5*width/2 - 0.3, person_ss_tres, width,
                    alpha=0.6, capsize=10, label='personal_ss_max_4_threshold')
    # Add text above bars
    for i, v in enumerate(person_ss_tres):
        # ax.text(i-3*width/2, v+0.02, f"{v:.2f}\n±{f1_scores_std[i]:.2f}", ha='center')
        ax.text(i-5*width/2 - 0.28, v+0.02, f"{v*100:.0f}", ha='center')

    rects1 = ax.bar(x - 3*width/2 - 0.25, accuracies_b[0], width,
                    alpha=0.6, capsize=10, label=thresholds[0])
    # Add text above bars
    for i, v in enumerate(accuracies_b[0]):
        # ax.text(i-3*width/2, v+0.02, f"{v:.2f}\n±{f1_scores_std[i]:.2f}", ha='center')
        ax.text(i-3*width/2 - 0.25, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of mean recall score
    rects2 = ax.bar(x - width/2 - 0.20, accuracies_b[1], width,
                    alpha=0.6, capsize=10, label=thresholds[1])
    # Add text above bars
    for i, v in enumerate(accuracies_b[1]):
        # ax.text(i-width/2, v+0.02, f"{v:.2f}\n±{recall_scores_std[i]:.2f}", ha='center')
        ax.text(i-width/2 - 0.20, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of mean precision score
    rects3 = ax.bar(x + width/2 - 0.15, accuracies_b[2], width,
                    alpha=0.6, capsize=10, label=thresholds[2])
    # Add text above bars
    for i, v in enumerate(accuracies_b[2]):
        # ax.text(i+width/2, v+0.02, f"{v:.2f}\n±{precisions_scores_std[i]:.2f}", ha='center')
        ax.text(i+width/2 - 0.15, v+0.02, f"{v*100:.0f}", ha='center')
        
    # Plot bar chart of multi-class accuracy
    rects6 = ax.bar(x + 3*width/2 - 0.1, accuracies_b[3], width, alpha=0.6, capsize=10,
                    label=thresholds[3])
    # Add text above bars
    for i, v in enumerate(accuracies_b[3]):
        ax.text(i+3*width/2 - 0.1, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of binary accuracy
    rects4 = ax.bar(x + 5*width/2 - 0.05, accuracies_b[4], width, alpha=0.6, capsize=10,
                    label=thresholds[4])
    # Add text above bars
    for i, v in enumerate(accuracies_b[4]):
        ax.text(i+ 5*width/2 - 0.05, v+0.02, f"{v*100:.0f}", ha='center')



    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('Binary Accuracy')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='best')
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()
    
def plot_ss_scors(accuracies_b, accuracies_b_0, accuracies_b_per_ss_, accuracies_b_per_ss_nax, models_names, path):
    # Generate heatmap of F1-score across all labels and models
    # Extract metrics for each label and model
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_names))
    width = 0.1
    
    rects1 = ax.bar(x - width/2 - 0.20, accuracies_b, width,
                    alpha=0.6, capsize=10, label='personal_ss_5_threshold')
    # Add text above bars
    for i, v in enumerate(accuracies_b):
        ax.text(i-width/2 - 0.2, v+0.02, f"{v*100:.0f}", ha='center')
        
    rects1 = ax.bar(x + width/2 - 0.15, accuracies_b_0, width,
                    alpha=0.6, capsize=10, label='personal_ss_max_4_threshold')
    # Add text above bars
    for i, v in enumerate(accuracies_b_0):
        ax.text(i+width/2 - 0.15, v+0.02, f"{v*100:.0f}", ha='center')

    rects1 = ax.bar(x + 3*width/2 - 0.1, accuracies_b_per_ss_, width,
                    alpha=0.6, capsize=10, label='personal_ss_no_q_threshold')
    # Add text above bars
    for i, v in enumerate(accuracies_b_per_ss_):
        ax.text(i+3*width/2 - 0.1, v+0.02, f"{v*100:.0f}", ha='center')

    # Plot bar chart of mean recall score
    rects2 = ax.bar(x + 5*width/2 - 0.05, accuracies_b_per_ss_nax, width,
                    alpha=0.6, capsize=10, label='personal_ss_no_q_max_threshold')
    # Add text above bars
    for i, v in enumerate(accuracies_b_per_ss_nax):
        # ax.text(i-width/2, v+0.02, f"{v:.2f}\n±{recall_scores_std[i]:.2f}", ha='center')
        ax.text(i+ 5*width/2 - 0.05, v+0.02, f"{v*100:.0f}", ha='center')



    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('Binary Accuracy')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()
    
def plot_sieamis_b_scors(models_names, accuracies_b, path):
    
    # Define bar plot
    fig, ax = plt.subplots(figsize=(25, 10))
    
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
    ax.set_ylim(0, 1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()

def evaluate_sieamis_classification(predictions, binary_ground_truth, real_class, threshold=0.5):
    
    predictions_list = [sublist[0] for sublist in predictions]
    classified_threshold_list = [1 if value > threshold else 0 for value in predictions_list]
    
    classified_max_ssq_list = []
    for i in range(0, len(predictions_list), 5):
        group = predictions_list[i:i+5]
        max_value = max(group)
        group_classification = [1 if value == max_value else 0 for value in group]
        classified_max_ssq_list.extend(group_classification)
    
    multiclass_ground_truth = []
    for _, ss in enumerate(real_class):
        multiclass_ground_truth.append(ss[0][0][-2:])
            
    accuracy_b = accuracy_score(binary_ground_truth, classified_max_ssq_list)
    conf_matrix_b = confusion_matrix(binary_ground_truth, classified_max_ssq_list)
    report_b = classification_report(binary_ground_truth, classified_max_ssq_list, output_dict=True)
    binary = [accuracy_b, conf_matrix_b, report_b]
    
    mlc_predictions = []
    real_class_lables = [rcl[-2:] for rc in real_class for rcl in rc[1]]
    for i in range(0, len(predictions_list), 5):
        
        group_predictions = predictions_list[i:i+5]
        group_labels = real_class_lables[i:i+5]
        
        group_max_index = max(range(len(group_predictions)), key=group_predictions.__getitem__)
        predicted_label = group_labels[group_max_index]
        mlc_predictions.append(predicted_label)
        
    accuracy_multi = accuracy_score(multiclass_ground_truth, mlc_predictions)
    conf_matrix_multi = confusion_matrix(multiclass_ground_truth, mlc_predictions)
    report_multi = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
    
    multi = [accuracy_multi, conf_matrix_multi, report_multi]

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
        for label in report[1][2].keys():
            if label.isdigit():
                f1_scores_per_report.append(report[1][2][label]['f1-score'])
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
        for label in report[1][2].keys():
            if label.isdigit():
                recall_scores_per_report.append(report[1][2][label]['recall'])
        recalls_scores.append(recall_scores_per_report)
        recall_scores_mean.append(np.mean(recall_scores_per_report))
        recall_scores_std.append(np.std(recall_scores_per_report))
        balanced_accuracy.append((1/n_classes) * sum(recall_scores_per_report))
        
    precisions_scores_mean = []
    precisions_scores_std = []
    precisions_scores = []
    for report in classifications_repo:
        precision_scores_per_report = []
        for label in report[1][2].keys():
            if label.isdigit():
                precision_scores_per_report.append(report[1][2][label]['precision'])
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