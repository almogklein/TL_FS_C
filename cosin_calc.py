from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import os, json
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


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
            # distance = cosine_similarity_numpy(emb1, emb2)
            # dis_sklear = cosine_similarity_sklearn(emb1, emb2)
            dis_scipy = cosine_similarity_scipy(emb1, emb2)
            
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

def evaluate_classification(cosine_distances, pairs):
    

    binary_ground_truth = [pair[0] for pair in pairs] 
    
    multiclass_ground_truth = []
    for i, pair in enumerate(pairs):
        if (i+1) % 5 == 0:
            multiclass_ground_truth.append(int(pair[1][-2:]))
    
    accuracies_b = []
    conf_matrices_b = []
    reports_b = []
    
    accuracies_mc = []
    conf_matrices_mc = []
    classifications_repo_mc = []
    mc_p = []
    for i in range(len(cosine_distances[0])):
        model_distances = [dist[i] for dist in cosine_distances]
        bc_predictions = [1 if distance > 0.5 else 0 for distance in model_distances]

        accuracy = accuracy_score(binary_ground_truth, bc_predictions)
        conf_matrix = confusion_matrix(binary_ground_truth, bc_predictions)
        report = classification_report(binary_ground_truth, bc_predictions, output_dict=True)

        accuracies_b.append(accuracy)
        conf_matrices_b.append(conf_matrix)
        reports_b.append(report)
        
        mc_pair = []
        p_i = []
        for i, dis in enumerate(model_distances):
            mc_pair.append(dis)
            if (i+1) % 5 == 0:
                p_i.append(model_distances.index(max(mc_pair)))
                mc_pair = []
                
        mlc_predictions = [int(pairs[index][2][-2:]) for index in p_i]
        mc_p.append(mlc_predictions)
        
        report = classification_report(multiclass_ground_truth, mlc_predictions, output_dict=True)
        accuracy = accuracy_score(multiclass_ground_truth, mlc_predictions)
        conf_matrix = confusion_matrix(multiclass_ground_truth, mlc_predictions)
        
        accuracies_mc.append(accuracy)
        conf_matrices_mc.append(conf_matrix)
        classifications_repo_mc.append(report)
        
    return classifications_repo_mc, accuracies_b, accuracies_mc, conf_matrices_mc, reports_b, mc_p 

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

def plot_scors(classifications_repo, models_names, accuracies_b, accuracies_mc, path):
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

    # Plot bar chart of mean F1-score
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
        
        
    # Plot bar chart of binary accuracy
    rects4 = ax.bar(x + 3*width/2 - 0.1, accuracies_b, width, alpha=0.6, capsize=10,
                    label='Binary Accuracy')
    # Add text above bars
    for i, v in enumerate(accuracies_b):
        ax.text(i+3*width/2 - 0.1, v+0.02, f"{v*100:.0f}", ha='center')

    # # Plot bar chart of multi-class accuracy
    # rects5 = ax.bar(x + 5*width/2 - 0.05, accuracies_mc, width, alpha=0.7, capsize=5,
    #                 label='Multi-class Accuracy')
    # # Add text above bars
    # for i, v in enumerate(accuracies_mc):
    #     ax.text(i+5*width/2 - 0.05, v+0.02, f"{v:.2f}", ha='center')
    
    # Plot bar chart of multi-class accuracy
    rects6 = ax.bar(x + 5*width/2 - 0.05, balanced_accuracy, width, alpha=0.6, capsize=10,
                    label='Multi-class balanced_acc')
    # Add text above bars
    for i, v in enumerate(balanced_accuracy):
        ax.text(i+5*width/2 - 0.05, v+0.02, f"{v*100:.0f}", ha='center')


    # Add labels, title and axis ticks
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend()
    plt.xticks(rotation=35)
    plt.savefig(path, bbox_inches='tight')

    plt.show()