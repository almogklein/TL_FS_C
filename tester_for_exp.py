import numpy as np
import data_maker_trainer_for_exp as dtm
import cosin_calc as cc
import util as u

ESC50_PATH = '/home/almogk/ESC-50-master'
CODE_REPO_PATH = '/home/almogk/FSL_TL_E_C'

class tester:
    
    def __init__(self, maker) -> None:
        
        self.test_support_set_num = maker.test_support_set_num
        self.query_c_num = maker.query_c_num
        self.query_c_size = maker.query_c_size
        self.id2label_map = maker.id2label_map 
        self.k_way = maker.k_way
        self.embeddings_full = maker.embeddings_full
        self.create_ss_treesh_cosin = None
        self.val_support_set_num = maker.val_support_set_num
        self.models_names = ['scratch rand pos', 'scratch T(ESC-35)', 'PT (ImagNet)', 
                             'PT (ImagNet) FT (ESC-35)', 'PT (ImagNet, AudioSet)', 'PT (ImagNet, AudioSet) FT (ESC-35)']
        
    def _infer(self, all_pairs= False, sim_create= False, make_mc_max_plot= False, gen_treshold= False, 
               make_ss_tresh= False, save_dic_plots = False, create_ss_treesh_cosin = False):
        
        self.create_ss_treesh_cosin = create_ss_treesh_cosin
        if all_pairs:
            pair_dic = u.make_all_pairs(CODE_REPO_PATH, self.test_support_set_num, self.query_c_num, self.query_c_size, self.k_way)
        
        save_file_path = CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/'
        
        test_pairs_q = u.read_json(save_file_path + f'/test/15000/75000_test_15000__1C_1PC_task_sets.json')
        test_pairs_no_q = u.read_json(save_file_path + f'/test/15000/150000_test_15000_ss_task_sets.json')
 
        pairs_q = u.read_json(save_file_path + f'/val/120/3000_val_120__5C_1PC_task_sets.json')
        pairs_no_q = u.read_json(save_file_path + f'/val/120/1200_val_120_ss_task_sets.json')

        if sim_create:
            
            cosine_distances = cc.calculate_cosine_distances(self.embeddings_full, pairs_q)
            cosine_distances_no_q = cc.calculate_cosine_distances(self.embeddings_full, pairs_no_q)
            
            cosine_distances_test = cc.calculate_cosine_distances(self.embeddings_full, test_pairs_q)
            cosine_distances_test_no_q = cc.calculate_cosine_distances(self.embeddings_full, test_pairs_no_q)
            
            u.write_json(save_file_path + f'/val/120/cos_sim_val_q.json', cosine_distances)
            u.write_json(save_file_path + f'/val/120/cos_sim_val_no_q.json', cosine_distances_no_q)
            
            u.write_json(save_file_path + f'/test/15000/cos_sim_test_q.json', cosine_distances_test)
            u.write_json(save_file_path + f'/test/15000/cos_sim_tesr_no_q.json', cosine_distances_test_no_q)
            
        else:
            cosine_distances = u.read_json(save_file_path + f'/val/120/cos_sim_val_q.json')
            cosine_distances_no_q = u.read_json(save_file_path + f'/val/120/cos_sim_val_no_q.json')
    
            cosine_distances_test = u.read_json(save_file_path + f'/test/15000/cos_sim_test_q.json')
            cosine_distances_test_no_q = u.read_json(save_file_path + f'/test/15000/cos_sim_tesr_no_q.json')

        if make_mc_max_plot:
            mc_p, balanced_accuracy_mc, accuracies_mc, reports_mc, conf_matrices_mc = cc.evaluate_classification_multiclass_closet_max([lst[:4] + [lst[5], lst[4]] for lst in cosine_distances_test], test_pairs_q)            
            cc.plot_scors(self.models_names, reports_mc, balanced_accuracy_mc, 0, save_file_path+f'/test/15000/scors_multic_max_5__________fix_________________.png')
        
        if gen_treshold:
            
            num_thresholds = 500
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
                
                if save_dic_plots:
                        
                    cc.calculate_statistics(cosine_distances, pairs_q, self.embeddings_full, save_file_path+'/stats.csv')
                    cc.plot_scors(self.models_names, reports_b, save_file_path+f'/__scors_{threshold}.png')
                    
                    for ii in range(len(self.models_names)):
                        print(f"{self.models_names[ii]}")
                        print(f"\n binary Accuracy: {accuracies_b[ii]}")
                        print(f"\n binary Confusion Matrix:\n{conf_matrices_b[ii]}")
                        print(f"\n binary Classification Report for:\n{reports_b[ii]}")
                        
                        print(f"\n Multiclass Accuracy: {accuracies_mc[ii]}")
                        print(f"\n Multiclass Confusion Matrix:\n{conf_matrices_mc[ii]}")
                        print(f"\n Multiclass Classification Report for:\n{reports_mc[ii]}")
                        
                        cc.plot_combined(conf_matrices_b[ii], reports_b[ii], self.models_names[ii], save_file_path+f'/{self.models_names[ii]}_cosine_binary_plot.png')
                    balanced_t.append(accuracies_b)
                    cc.plot_scors(self.models_names, reports_b, save_file_path+f'/__scors_{threshold}.png')
            
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
            
            cc.plot_b_scors(reports_b_TEST, eer_values, balanced_accuracies_b_TEST, thresholds_list, self.models_names, save_file_path+f'/balance_acc_b_closet_test________fix.png', 'balance_acc')
            
        if make_ss_tresh:
            
            save_file1 = save_file_path + f'/val/120/ss_personal_param_q_val.json'
            save_file2 = save_file_path + f'/val/120/ss_personal_param_no_q_val.json'
            
            save_file3 = save_file_path + f'/test/15000/ss_personal_param_q_test.json'
            save_file4 = save_file_path + f'/test/15000/ss_personal_param_no_q_test.json'
            
            if self.create_ss_treesh_cosin:
                perso_ss_param_q_val = u.make_perso_ss_param(cosine_distances, save_file1)                
                perso_ss_param_no_q_val = u.make_perso_ss_param_no_q(cosine_distances_no_q, save_file2)
                
                perso_ss_param_q_test = u.make_perso_ss_param(cosine_distances_test, save_file3)                
                perso_ss_param_no_q_test = u.make_perso_ss_param_no_q(cosine_distances_test_no_q, save_file4)
                
            else:
                perso_ss_param_q_val = u.read_json(save_file1)
                perso_ss_param_no_q_val = u.read_json(save_file2)
                
                perso_ss_param_q_test = u.read_json(save_file3)
                perso_ss_param_no_q_test = u.read_json(save_file4)
            
            tresh_sig_const = np.linspace(0, 10, num=1000)
            tresh_alfa_const = np.linspace(0, 10, num=1000)
            
            err_list_, acc_list_ = [], []
            acc_list_5_mss, acc_list_5_mad, acc_list_4_mss, acc_list_4_mad, acc_list_10_mss, acc_list_10_mass_ = [[] for _ in range(6)]
            err_list_5_mss, err_list_5_mad, err_list_4_mss, err_list_4_mad, err_list_10_mss, err_list_10_mass_ = [[] for _ in range(6)]
            all_tresh_ = []
            ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_q]
            binary_ground_truth_val = [pair[0] for pair in pairs_q]
            
            for index_const, (sig, alf) in enumerate(zip(tresh_sig_const, tresh_alfa_const)):
                ss_tresholds_all, ss_tresholds_all_0, ss_tresholds_no_q_max_sig_std, ss_tresholds_no_q_mean_sig_std, ss_tresholds_all_max_alfa_diff, ss_tresholds_all_0_max_alfa_diff = [[] for _ in range(6)]
                for i in range(len(self.models_names)):
                    
                    ss_tresholds_all.append([mad - sig*std for mad, std in zip(perso_ss_param_q_val['max'][i], perso_ss_param_q_val['std_all'][i])])                    
                    ss_tresholds_all_max_alfa_diff.append([p95 - alf*diff_ for p95, diff_ in zip(perso_ss_param_q_val['max'][i], perso_ss_param_q_val['f_s_dif'][i])])

                    ss_tresholds_all_0.append([mean + sig*std for mean, std in zip(perso_ss_param_q_val['max0'][i], perso_ss_param_q_val['std_0'][i])])
                    ss_tresholds_all_0_max_alfa_diff.append([max_ + alf*diff_ for max_, diff_ in zip(perso_ss_param_q_val['max0'][i], perso_ss_param_q_val['f_s_dif'][i])])
                    
                    ss_tresholds_no_q_mean_sig_std.append([value for value in [mean[0] + sig*std[0] for mean, std in zip(perso_ss_param_no_q_val['max'][i], perso_ss_param_no_q_val['std_all'][i])] for _ in range(self.query_c_num[1])])
                    ss_tresholds_no_q_max_sig_std.append([max_[0] + sig*std for max_, std in zip([value for val in perso_ss_param_no_q_val['max'][i] for value in [val] * self.query_c_num[1]], perso_ss_param_q_val['f_s_dif'][i])])

                bc_p_mss, accuracies_b_mss, reports_b_mss, conf_matrices_b_mss, eer_mss, bala_acc_mss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)
                bc_p_MAD, accuracies_b_mad, reports_b_mad, _, eer_mad, bala_acc_mad = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_max_alfa_diff, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)

                bc_p_0_mss, accuracies_b_0_mss, reports_b_0_mss, _, eer_0_mss, bala_acc_0_mss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_0, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)
                bc_p_0_MAD, accuracies_b_0_mad, reports_b_0_mad, _, eer_0_mad, bala_acc_0_mad = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_all_0_max_alfa_diff, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)
                
                bc_p_ss, accuracies_b_per_ss, reports_b_ss, _, eer_ss, bala_acc_ss = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_no_q_mean_sig_std, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)
                bc_p_0_ss, accuracies_b_per_ss_max, reports_b_ss_max_, _, eer_ss_, bala_acc_ss_max = cc.evaluate_classification_per_ss_B(cosine_distances, pairs_q, ss_tresholds_no_q_max_sig_std, self.val_support_set_num[0], self.k_way[0], True, self.query_c_num[1], self.query_c_size[0], binary_ground_truth_val)
                
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
            sig_alfa_list = [sig_alfa_acc, sig_alfa_eer]
            
            ss_true_labels_test = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_q]
            binary_ground_truth_test = [pair[0] for pair in test_pairs_q]
            for  ac_eer_i, sig_alfa in enumerate(sig_alfa_list):
                
                ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_max_sig_std_test, ss_tresholds_no_q_mean_sig_std_test, ss_tresholds_all_max_alfa_diff_test, ss_tresholds_all_0_max_alfa_diff_test = [[] for _ in range(6)]
                for i in range(len(self.models_names)):
                    
                    ss_tresholds_all_test.append([mad + sig_alfa[0][i]*std for mad, std in zip(perso_ss_param_q_test['mean_all'][i], perso_ss_param_q_test['std_all'][i])])                    
                    ss_tresholds_all_max_alfa_diff_test.append([p95 - sig_alfa[1][i]*diff_ for p95, diff_ in zip(perso_ss_param_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])
                    
                    ss_tresholds_all_0_test.append([mean + sig_alfa[2][i]*std for mean, std in zip(perso_ss_param_q_test['mean_0'][i], perso_ss_param_q_test['std_0'][i])])
                    ss_tresholds_all_0_max_alfa_diff_test.append([max_ + sig_alfa[3][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['max0'][i], perso_ss_param_q_test['f_s_dif'][i])])
                    
                    ss_tresholds_no_q_mean_sig_std_test.append([mean[0] + sig_alfa[4][i]*std[0] for mean, std in zip(perso_ss_param_no_q_test['mean_all'][i], perso_ss_param_no_q_test['std_all'][i])])
                    ss_tresholds_no_q_max_sig_std_test.append([max_[0] + sig_alfa[5][i]*std for max_, std in zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])

                bc_p_mss, accuracies_b_test5_mss, reports_b_mss, conf_matrices_b_mss, eer_test5_mss, BA_test5_mss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
                bc_p_MAD, accuracies_b_test5_mad, reports_b_mad, conf_matrices_b_mss_, eer_test5_mad, BA_test5_mad = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_max_alfa_diff_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
                
                bc_p_0_mss, accuracies_b_test4_0_mss, reports_b_0_mss, conf_matrices_b_mss__, eer_test4_0_mss, BA_test4_0_mss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_0_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
                bc_p_0_MAD, accuracies_b_test4_0_mad, reports_b_0_mad, conf_matrices_b_mss____, eer_test4_0_mad, BA_test4_0_mad = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_all_0_max_alfa_diff_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
                
                bc_p_ss, accuracies_b_test10, reports_b_ss, conf_matrices_b_mss__________________, eer_test10_ss, BA_test10_ss = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_no_q_mean_sig_std_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
                bc_p_0_ss, accuracies_b_test10_max, reports_b_ss_max_, conf_matrices_b_mss_____________________________, eer_test10_ss_, BA_test10_ss_ = cc.evaluate_classification_per_ss_B(cosine_distances_test, test_pairs_q, ss_tresholds_no_q_max_sig_std_test, self.test_support_set_num[0], self.k_way[0], True, self.query_c_num[0], self.query_c_size[0], binary_ground_truth_test)
            
                cc.plot_ss_scors([BA_test5_mss, BA_test5_mad], 
                                 [BA_test4_0_mss, BA_test4_0_mad], 
                                 [BA_test10_ss, BA_test10_ss_], sig_alfa, self.models_names, save_file_path+f'/{ac_eer_i}____fix________________ori___________scors_tresholds_test.png')

    def _infer_openset(self, open_set_personal_CAT_tresh = False, open_set_personal_tresh = False, open_set_fix_tresh = False, 
                       create_ = False, take_or_create = True):
        
        save_file_path = CODE_REPO_PATH + f'/data/FSL_SETS/5w_1s_shot/'
        
        test_pairs_openset = u.read_json(save_file_path + f'/test/15000/75000_test_15000_1C_1PC_task_sets_openset.json')
        test_pairs_no_q = u.read_json(save_file_path + f'/test/15000/150000_test_15000_ss_task_sets.json')
        
        pairs_openset = u.read_json(save_file_path + f'/val/120/3000_val_120_5C_1PC_task_sets_openset.json')        
        pairs_no_q = u.read_json(save_file_path + f'/val/120/1200_val_120_ss_task_sets.json')

        if take_or_create:
            cosine_distances_openset = u.read_json(save_file_path + f'/val/120/cosin_openset_val.json')
            cosine_distances_no_q = u.read_json(save_file_path + f'/val/120/cos_sim_val_no_q.json')
            
            test_cosine_distances_openset = u.read_json(save_file_path + f'/test/15000/cosin_openset_test.json')        
            cosine_distances_test_no_q = u.read_json(save_file_path + f'/test/15000/cos_sim_tesr_no_q.json')
        else:    
            
            cosine_distances_openset = cc.calculate_cosine_distances(self.embeddings_full, pairs_openset)
            u.write_json(save_file_path + f'/val/120/cosin_openset_val.json', cosine_distances_openset)
            
            test_cosine_distances_openset = cc.calculate_cosine_distances(self.embeddings_full, test_pairs_openset)
            u.write_json(save_file_path + f'/test/15000/cosin_openset_test.json', test_cosine_distances_openset)
            
            cosine_distances_no_q = cc.calculate_cosine_distances(self.embeddings_full, pairs_no_q)
            u.write_json(save_file_path + f'/val/120/cos_sim_val_no_q.json', cosine_distances_no_q)
            
            cosine_distances_test_no_q = cc.calculate_cosine_distances(self.embeddings_full, test_pairs_no_q)
            u.write_json(save_file_path + f'/test/15000/cos_sim_tesr_no_q.json', cosine_distances_test_no_q)
                
        save_file1 = save_file_path + f'/val/120/ss_personal_param_q_openset_val.json'
        save_file2 = save_file_path + f'/val/120/ss_personal_param_no_q_val.json'
        
        save_file3 = save_file_path + f'/test/15000/ss_personal_param_q_openset_test.json'
        save_file4 = save_file_path + f'/test/15000/ss_personal_param_no_q_test.json'
        if create_:
            perso_ss_param_q_val = u.make_perso_ss_param(cosine_distances_openset, save_file1)                
            perso_ss_param_no_q_val = u.make_perso_ss_param_no_q(cosine_distances_no_q, save_file2)
            
            perso_ss_param_q_test = u.make_perso_ss_param(test_cosine_distances_openset, save_file3)                
            perso_ss_param_no_q_test = u.make_perso_ss_param_no_q(cosine_distances_test_no_q, save_file4)
            
        else:            
            perso_ss_param_q_val = u.read_json(save_file1)        
            perso_ss_param_no_q_val = u.read_json(save_file2)            
        
            perso_ss_param_q_test = u.read_json(save_file3)
            perso_ss_param_no_q_test = u.read_json(save_file4)            
            
        if open_set_fix_tresh:
            
            thresholds = cc.calculate_fix_threshold_openset(cosine_distances_openset, [pair[0] for pair in pairs_openset])

            # Perform multiclass & binary  classification
            ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
            multiclass_predictions, binary_predictions = cc.multiclass_binary_openset_classification(test_cosine_distances_openset, thresholds, ss_true_labels)
            multiclass_predictions_from_start = []
            for i in range(len(multiclass_predictions)):
                multiclass_predictions_from_start.append([self.id2label_map[int(cla[0])] for cla in multiclass_predictions[i][-1]])
            
            binary_ground_truth = [pair[0] for pair in test_pairs_openset]
            multi_ground_truth = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
            multi_ground_truth_from_start = [cla if cla == 'unknown' else self.id2label_map[int(cla)] for cla in multi_ground_truth]
            
            accuracies_b, reports_b, conf_matrices_b, binary_bala_acc, accuracies_m, reports_m, conf_matrices_m, mc_bala_acc, class_labels = cc.evaluate_classification_openset(multiclass_predictions_from_start, binary_predictions, multi_ground_truth_from_start, binary_ground_truth)
            
            tresh_fin = [tr[0] for tr in thresholds]
            cc.plot_scors(self.models_names, reports_m, mc_bala_acc, tresh_fin, save_file_path+f'/bala_acc________________scors_mc_OPENSET__.png')
            cc.plot_confusion_matrices(conf_matrices_m, save_file_path, class_labels, single_plot=True)
        
        if open_set_personal_tresh:
            
            ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_openset]
            binary_ground_truth_val = [pair[0] for pair in pairs_openset]
            multi_ground_truth_num_val = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in pairs_openset][::5]
            multi_ground_truth_cat_val = [cla if cla == 'unknown' else self.id2label_map[int(cla)] for cla in multi_ground_truth_num_val]
            class_labels_val = list(set(multi_ground_truth_cat_val))
            class_labels_val.remove('unknown')
            class_labels_val.append('unknown')
            
            thresholds_param = cc.calculate_pers_threshold_openset(cosine_distances_openset, cosine_distances_no_q, [pairs_openset, pairs_no_q], [perso_ss_param_no_q_val, perso_ss_param_q_val], self.models_names, self.val_support_set_num, self.k_way, self.query_c_num, self.query_c_size, ss_true_labels_val, self.id2label_map, binary_ground_truth_val, multi_ground_truth_cat_val, class_labels_val)
                    
            binary_ground_truth = [pair[0] for pair in test_pairs_openset]  
            multi_ground_truth_num = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
            multi_ground_truth_cat = [cla if cla == 'unknown' else self.id2label_map[int(cla)] for cla in multi_ground_truth_num]
            class_labels = list(set(multi_ground_truth_cat))
            class_labels.remove('unknown')
            class_labels.append('unknown')
            ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
            
            for  ac_eer_i, sig_alfa in enumerate(thresholds_param):
                
                ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_max_sig_std_test, ss_tresholds_no_q_mean_sig_std_test, ss_tresholds_all_max_alfa_diff_test, ss_tresholds_all_0_max_alfa_diff_test = [[] for _ in range(6)]
                for i in range(len(self.models_names)):
                    
                    ss_tresholds_all_test.append([mean + sig_alfa[0][i]*std for mean, std in zip(perso_ss_param_q_test['MAD'][i], perso_ss_param_q_test['std_all'][i])])                    
                    ss_tresholds_all_max_alfa_diff_test.append([max_ - sig_alfa[1][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])
                    
                    ss_tresholds_all_0_test.append([mean + sig_alfa[2][i]*std for mean, std in zip(perso_ss_param_q_test['MAD_0'][i], perso_ss_param_q_test['std_0'][i])])
                    ss_tresholds_all_0_max_alfa_diff_test.append([max_ + sig_alfa[3][i]*diff_ for max_, diff_ in zip(perso_ss_param_q_test['max0'][i], perso_ss_param_q_test['f_s_dif'][i])])
                    
                    ss_tresholds_no_q_mean_sig_std_test.append([mean[0] + sig_alfa[4][i]*std[0] for mean, std in zip(perso_ss_param_no_q_test['MAD'][i], perso_ss_param_no_q_test['std_all'][i])])
                    ss_tresholds_no_q_max_sig_std_test.append([max_[0] + sig_alfa[5][i]*std for max_, std in zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_q_test['f_s_dif'][i])])
                
                multiclass_prediction_5_mss, binary_predictions_5_mss, incd_all_5_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                multiclass_prediction_5_mad, binary_predictions_5_mad, incd_all_5_mad = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_max_alfa_diff_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                
                multiclass_prediction_4_mss, binary_predictions_4_mss, incd_all_4_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                multiclass_prediction_4_mad, binary_predictions_4_mad, incd_all_4_mad = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_max_alfa_diff_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                
                multiclass_prediction_10_mss, binary_predictions_10_mss, incd_all_10_mss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_mean_sig_std_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
                multiclass_prediction_10_maxss, binary_predictions_10_maxss, incd_all_10_maxss = cc.classification_per_ss_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_max_sig_std_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
        
                cc.plot_ss_scors([[elem[-1] for elem in incd_all_5_mss], [elem[-1] for elem in incd_all_5_mad]], 
                                [[elem[-1] for elem in incd_all_4_mss], [elem[-1] for elem in incd_all_4_mad]], 
                                [[elem[-1] for elem in incd_all_10_mss], [elem[-1] for elem in incd_all_10_maxss]], 
                                sig_alfa, self.models_names, save_file_path+f'/{ac_eer_i}_scors_mc_tresholds_openset_test_______________finall________________.png')
                
                # cc.plot_scors(models_names, incd_all_10_maxss, incd_all_10_maxss, save_file_path+f'/{ac_eer_i}_scors_mc_OPENSET______.png')
                # cc.plot_confusion_matrices(conf_matrices_m, save_file_path, class_labels, single_plot=False)        
        
        if open_set_personal_CAT_tresh:
            
            ss_true_labels_val = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in pairs_openset]
            binary_ground_truth_val = [pair[0] for pair in pairs_openset]
            multi_ground_truth_num_val = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in pairs_openset][::5]
            multi_ground_truth_cat_val = [cla if cla == 'unknown' else self.id2label_map[int(cla)] for cla in multi_ground_truth_num_val]
            class_labels_val = list(set(multi_ground_truth_cat_val))
            class_labels_val.remove('unknown')
            class_labels_val.append('unknown')
            
            sig_max, sig_mean, sig_median, ind_max = cc.calculate_pers_CAT_threshold_openset(cosine_distances_openset, cosine_distances_no_q, [pairs_openset, pairs_no_q], [perso_ss_param_no_q_val, perso_ss_param_q_val], self.models_names, self.val_support_set_num, self.k_way, self.query_c_num, self.query_c_size, ss_true_labels_val, self.id2label_map, binary_ground_truth_val, multi_ground_truth_cat_val, class_labels_val)
                    
            binary_ground_truth = [pair[0] for pair in test_pairs_openset]  
            multi_ground_truth_num = [pair[1][-2:] if pair[1][-2:].isdigit() else pair[1] for pair in test_pairs_openset][::5]
            multi_ground_truth_cat = [cla if cla == 'unknown' else self.id2label_map[int(cla)] for cla in multi_ground_truth_num]
            class_labels = list(set(multi_ground_truth_cat))
            class_labels.remove('unknown')
            class_labels.append('unknown')
            ss_true_labels = [pair[2][-2:] if pair[2][-2:].isdigit() else pair[2] for pair in test_pairs_openset]
            
            ss_tresholds_all_test, ss_tresholds_all_0_test, ss_tresholds_no_q_mean_sig_std_test = [[] for _ in range(3)]
            for i in range(len(self.models_names)):
                for cat_ind in range(1, self.k_way[0]+1):
                    ss_tresholds_all_test.append([mean[cat_ind] + sig_max[i]*std[cat_ind] for  _, (mean, std) in enumerate(zip(perso_ss_param_no_q_test['mean_all'][i], perso_ss_param_no_q_test['std_all'][i]))])                    
                    ss_tresholds_all_0_test.append([Max[cat_ind] + sig_mean[i]*std[cat_ind] for  _, (Max, std) in enumerate(zip(perso_ss_param_no_q_test['max'][i], perso_ss_param_no_q_test['std_all'][i]))])
                    ss_tresholds_no_q_mean_sig_std_test.append([mad[cat_ind] + sig_median[i]*std[cat_ind] for  _, (mad, std) in enumerate(zip(perso_ss_param_no_q_test['MAD'][i], perso_ss_param_no_q_test['std_all'][i]))])
            
            multiclass_prediction_5_mss, binary_predictions_5_mss, incd_all_5_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], self.query_c_size[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
            multiclass_prediction_4_mss, binary_predictions_4_mss, incd_all_4_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_all_0_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], self.query_c_size[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
            multiclass_prediction_10_mss, binary_predictions_10_mss, incd_all_10_mss = cc.classification_per_ss_CAT_mc(test_cosine_distances_openset, test_pairs_openset, ss_tresholds_no_q_mean_sig_std_test, self.test_support_set_num[0], self.k_way[0], self.query_c_num[0], self.query_c_size[0], ss_true_labels, self.id2label_map, binary_ground_truth, multi_ground_truth_cat, class_labels)
    
            cc.plot_ss_scors([elem[-1] for elem in incd_all_5_mss], [elem[-1] for elem in incd_all_4_mss], [elem[-1] for elem in incd_all_10_mss], [sig_max, sig_mean, sig_median], self.models_names, save_file_path+f'/PER_CAT_______scors_mc_tresholds_openset_test.png')
            # cc.plot_scors(models_names, incd_all_10_maxss, incd_all_10_maxss, save_file_path+f'/{ac_eer_i}_scors_mc_OPENSET______.png')
            # cc.plot_confusion_matrices(conf_matrices_m, save_file_path, class_labels, single_plot=False)

