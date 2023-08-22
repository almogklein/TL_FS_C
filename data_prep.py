# ######################## Installations ###############################
import os
import wget
import json
import random
import zipfile
import numpy as np
import pandas as pd
# #####################################################################

def get_data_for_ast_model_kfold(data_path, code_path, train_class_names, test_class_names, val_class_names):
    """
    This function generates the data files for the AST model.
    Arguments:
    - data_path (str): The path to the data directory.
    - code_path (str): The path to the code directory.
    - train_class_names (list): The names of the classes in the training set.
    - test_class_names (list): The names of the classes in the test set.
    - val_class_names (list): The names of the classes in the validation set.
    Returns:
    - None      
    """
    
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]
    
    def divide_dict(input_dict, list1, list2, list3):
        
        """
        Divides a dictionary into 3 new dictionaries based on the keys in the provided lists.
        Arguments:
        - input_dict (dict): The dictionary to divide.
        - list1 (list): The keys to include in the first output dictionary.
        - list2 (list): The keys to include in the second output dictionary.
        - list3 (list): The keys to include in the third output dictionary.
        Returns:
        - tuple: Three dictionaries, each with the keys from the corresponding list.
        """
        dict1 = {k: v for k, v in input_dict.items() if k in list1}
        dict2 = {k: v for k, v in input_dict.items() if k in list2}
        dict3 = {k: v for k, v in input_dict.items() if k in list3}
        
        return dict1, dict2, dict3
    
    # downlooad esc50
    # dataset provided in https://github.com/karolpiczak/ESC-50
    if os.path.exists(data_path) == False:
        
        esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
        wget.download(esc50_url, out='/home/almogk')
        with zipfile.ZipFile('/home/almogk/ESC-50-master.zip', 'r') as zip_ref:
            zip_ref.extractall('/home/almogk/')
        os.remove('/home/almogk/ESC-50-master.zip')
        # convert the audio to 16kHz
        os.mkdir(data_path + '/audio_16k')
        audio_list = get_immediate_files(data_path + '/audio')
        for audio in audio_list:
            print('sox ' + data_path + '/audio/' + audio + ' -r 16000 ' + data_path + '/audio_16k/' + audio)
            os.system('sox ' + data_path + '/audio/' + audio + ' -r 16000 ' + data_path + '/audio_16k/' + audio)
    

    # generate an empty directory to save json files
    if os.path.exists(code_path + '/data_kfold') == False:
        os.mkdir(code_path + '/data_kfold')
        os.mkdir(code_path + '/data_kfold/data_files')
        os.mkdir(code_path + '/data_kfold/train_datafile')
        os.mkdir(code_path + '/data_kfold/train_datafile/train_cla_datafile')
        os.mkdir(code_path + '/data_kfold/train_datafile/train_fsl_datafile')
        os.mkdir(code_path + '/data_kfold/test_datafile')
        os.mkdir(code_path + '/data_kfold/val_datafile')
        
        label_set = np.loadtxt(code_path + '/esc_class_labels_indices.csv', delimiter=',', dtype='str')
        label_map = {}
        for i in range(1, len(label_set)):
            label_map[eval(label_set[i][2])] = label_set[i][0]
        
        esc_class_labels_indices = pd.read_csv(code_path + '/esc_class_labels_indices.csv')
        meta = np.loadtxt((data_path + '/meta/esc50.csv'), delimiter=',', dtype='str', skiprows=1)
        
        class_look_all_folds = {}
        for k in range(len(test_class_names)):
            
            # Read the CSV file
            esc_train_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(train_class_names[k])]
            esc_test_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(test_class_names[k])]
            esc_val_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(val_class_names[k])]
            
            class_lookup_train = {}
            for index, row in esc_class_labels_indices.iterrows():
                if row['display_name'] in train_class_names[k]:
                    class_lookup_train[str(int(row['mid'][3:][-2:]))] = train_class_names[k].index(row['display_name'])
            
            # Write the three CSV files to separate files
            esc_train_class_labels_indices['index'] = esc_train_class_labels_indices['index'].astype(str).map(class_lookup_train)
            
            esc_train_class_labels_indices.to_csv(code_path + f'/esc_train_class_labels_indices_fold_{k}.csv', index=False)
            esc_test_class_labels_indices.to_csv(code_path + f'/esc_test_class_labels_indices_fold_{k}.csv', index=False)
            esc_val_class_labels_indices.to_csv(code_path + f'/esc_val_class_labels_indices_fold_{k}.csv', index=False)
            
            # Get the index of rows that belong to the validation set
            val_index = np.isin(meta[:, 3], val_class_names[k])
            # Get the index of rows that belong to the test set
            test_index = np.isin(meta[:, 3], test_class_names[k])
            # Get the index of rows that belong to the training set
            train_index = np.isin(meta[:, 3], train_class_names[k])

            # Create the new objects with the corresponding rows
            meta_val = meta[val_index, :]
            meta_test = meta[test_index, :]
            meta_train = meta[train_index, :]
            wav_train_list = []
            wav_train_list_35 = []
            wav_test_list = []
            wav_val_list = []
            wav_list = []
            
            for i in range(0, len(meta)):
                cur_label = label_map[meta[i][3]]
                cur_path = meta[i][0]
                cur_dict = {"wav": data_path + '/audio_16k/' + cur_path, "labels": '/m/07rwj' + cur_label.zfill(2)}
                wav_list.append(cur_dict)
            
            for i in range(0, len(meta_train)):
                cur_label_train = label_map[meta_train[i][3]]
                cur_path_train = meta_train[i][0]
                cur_fold_train = int(meta_train[i][1])
                
                cur_dict_train = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + cur_label_train.zfill(2)}
                cur_dict_train_35 = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + str(class_lookup_train[cur_label_train]).zfill(2)}
                wav_train_list.append(cur_dict_train)
                wav_train_list_35.append(cur_dict_train_35)
                
            for i in range(0, len(meta_test)):
                cur_label_test = label_map[meta_test[i][3]]
                cur_path_test = meta_test[i][0]
                # cur_fold_test = int(meta_test[i][1])
                # /m/07rwj is just a dummy prefix
                cur_dict_test = {"wav": data_path + '/audio_16k/' + cur_path_test, "labels": '/m/07rwj' + cur_label_test.zfill(2)}
                wav_test_list.append(cur_dict_test)
            
            for i in range(0, len(meta_val)):
                cur_label_val = label_map[meta_val[i][3]]
                cur_path_val = meta_val[i][0]
                cur_dict_val = {"wav": data_path + '/audio_16k/' + cur_path_val, "labels": '/m/07rwj' + cur_label_val.zfill(2)}
                wav_val_list.append(cur_dict_val)
            
            with open(code_path + f'/data_kfold/data_files/data_{k}.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
                
            with open(code_path + f'/data_kfold/train_datafile/train_fsl_datafile/esc_fsl_train_data_{k}.json', 'w') as f:
                json.dump({'data': wav_train_list}, f, indent=1)
            
            with open(code_path + f'/data_kfold/train_datafile/train_fsl_datafile/35_esc_fsl_train_data_{k}.json', 'w') as f:
                json.dump({'data': wav_train_list_35}, f, indent=1)
            
            with open(code_path + f'/data_kfold/test_datafile/esc_fsl_test_data_{k}.json', 'w') as f:
                json.dump({'data': wav_test_list}, f, indent=1)
                    
            with open(code_path + f'/data_kfold/val_datafile/esc_fsl_val_data_{k}.json', 'w') as f:
                json.dump({'data': wav_val_list}, f, indent=1)
                
        
            for fold in range(1, 6):
                
                Train_eval_wav_list = []
                Train_train_wav_list = []
                Train_eval_wav_list_35 = []
                Train_train_wav_list_35 = []

                for i in range(0, len(meta_train)):
                    cur_label_train = label_map[meta_train[i][3]]
                    cur_path_train = meta_train[i][0]
                    cur_fold_train = int(meta_train[i][1])
                    
                    # /m/07rwj is just a dummy prefix
                    cur_dict_train = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + cur_label_train.zfill(2)}
                    cur_dict_train_35 = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + str(class_lookup_train[cur_label_train]).zfill(2)}
                    
                    if cur_fold_train == fold:
                        Train_eval_wav_list.append(cur_dict_train)
                        Train_eval_wav_list_35.append(cur_dict_train_35)
                    else:
                        Train_train_wav_list.append(cur_dict_train)
                        Train_train_wav_list_35.append(cur_dict_train_35)
                print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(Train_train_wav_list), len(Train_eval_wav_list)))
                
                with open(code_path + f'/data_kfold/train_datafile/train_cla_datafile/esc_train_data_{fold}_all_{k}.json', 'w') as f:
                    json.dump({'data': Train_train_wav_list}, f, indent=1)

                with open(code_path + f'/data_kfold/train_datafile/train_cla_datafile/esc_eval_data__{fold}_all_{k}.json', 'w') as f:
                    json.dump({'data': Train_eval_wav_list}, f, indent=1)

                with open(code_path + f'/data_kfold/train_datafile/train_cla_datafile/35_esc_train_data__{fold}_all_{k}.json', 'w') as f:
                    json.dump({'data': Train_train_wav_list_35}, f, indent=1)

                with open(code_path + f'/data_kfold/train_datafile/train_cla_datafile/35_esc_eval_data__{fold}_all_{k}.json', 'w') as f:
                    json.dump({'data': Train_eval_wav_list_35}, f, indent=1)
                
            class_look_all_folds[k] = class_lookup_train
            print(f'Finished ESC-50 Preparation for fold {k}')
        return class_look_all_folds    

def get_data_for_ast_model(data_path, code_path, train_class_names, test_class_names, val_class_names, class_map):
    """
    This function generates the data files for the AST model.
    Arguments:
    - data_path (str): The path to the data directory.
    - code_path (str): The path to the code directory.
    - train_class_names (list): The names of the classes in the training set.
    - test_class_names (list): The names of the classes in the test set.
    - val_class_names (list): The names of the classes in the validation set.
    Returns:
    - None      
    """
    
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]
    
    def divide_dict(input_dict, list1, list2, list3):
        
        """
        Divides a dictionary into 3 new dictionaries based on the keys in the provided lists.
        Arguments:
        - input_dict (dict): The dictionary to divide.
        - list1 (list): The keys to include in the first output dictionary.
        - list2 (list): The keys to include in the second output dictionary.
        - list3 (list): The keys to include in the third output dictionary.
        Returns:
        - tuple: Three dictionaries, each with the keys from the corresponding list.
        """
        dict1 = {k: v for k, v in input_dict.items() if k in list1}
        dict2 = {k: v for k, v in input_dict.items() if k in list2}
        dict3 = {k: v for k, v in input_dict.items() if k in list3}
        
        return dict1, dict2, dict3
    
    # downlooad esc50
    # dataset provided in https://github.com/karolpiczak/ESC-50
    if os.path.exists(data_path) == False:
        
        esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
        wget.download(esc50_url, out='/home/almogk')
        with zipfile.ZipFile('/home/almogk/ESC-50-master.zip', 'r') as zip_ref:
            zip_ref.extractall('/home/almogk/')
        os.remove('/home/almogk/ESC-50-master.zip')
        # convert the audio to 16kHz
        os.mkdir(data_path + '/audio_16k')
        audio_list = get_immediate_files(data_path + '/audio')
        for audio in audio_list:
            print('sox ' + data_path + '/audio/' + audio + ' -r 16000 ' + data_path + '/audio_16k/' + audio)
            os.system('sox ' + data_path + '/audio/' + audio + ' -r 16000 ' + data_path + '/audio_16k/' + audio)
    

    # generate an empty directory to save json files
    if os.path.exists(code_path + '/data') == False:
        os.mkdir(code_path + '/data')
        os.mkdir(code_path + '/data/data_files')
        os.mkdir(code_path + '/data/train_datafile')
        os.mkdir(code_path + '/data/train_datafile/train_cla_datafile')
        os.mkdir(code_path + '/data/train_datafile/train_fsl_datafile')
        os.mkdir(code_path + '/data/test_datafile')
        os.mkdir(code_path + '/data/val_datafile')
        
        label_set = np.loadtxt(code_path + '/esc_class_labels_indices.csv', delimiter=',', dtype='str')
        label_map = {}
        for i in range(1, len(label_set)):
            label_map[eval(label_set[i][2])] = label_set[i][0]

        # train_label_map, test_label_map, val_label_map = divide_dict(label_map, train_class_names, test_class_names, val_class_names)
        
        # Read the CSV file
        esc_class_labels_indices = pd.read_csv('/home/almogk/FSL_TL_E_C/esc_class_labels_indices.csv')
        esc_train_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(train_class_names)]
        esc_test_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(test_class_names)]
        esc_val_class_labels_indices = esc_class_labels_indices[esc_class_labels_indices['display_name'].isin(val_class_names)]
        
        # Write the three CSV files to separate files
        esc_train_class_labels_indices['index'] = esc_train_class_labels_indices['index'].astype(str).map(class_map)
        
        esc_train_class_labels_indices.to_csv('/home/almogk/FSL_TL_E_C/esc_train_class_labels_indices.csv', index=False)
        esc_test_class_labels_indices.to_csv('/home/almogk/FSL_TL_E_C/esc_test_class_labels_indices.csv', index=False)
        esc_val_class_labels_indices.to_csv('/home/almogk/FSL_TL_E_C/esc_val_class_labels_indices.csv', index=False)
        
        meta = np.loadtxt((data_path + '/meta/esc50.csv'), delimiter=',', dtype='str', skiprows=1)
        # Get the index of rows that belong to the validation set
        val_index = np.isin(meta[:, 3], val_class_names)
        # Get the index of rows that belong to the test set
        test_index = np.isin(meta[:, 3], test_class_names)
        # Get the index of rows that belong to the training set
        train_index = np.isin(meta[:, 3], train_class_names)

        # Create the new objects with the corresponding rows
        meta_val = meta[val_index, :]
        meta_test = meta[test_index, :]
        meta_train = meta[train_index, :]
        wav_train_list = []
        wav_train_list_35 = []
        wav_test_list = []
        wav_val_list = []
        wav_list = []
        
        for i in range(0, len(meta)):
            cur_label = label_map[meta[i][3]]
            cur_path = meta[i][0]
            # cur_fold = int(meta[i][1])
            # /m/07rwj is just a dummy prefix
            cur_dict = {"wav": data_path + '/audio_16k/' + cur_path, "labels": '/m/07rwj' + cur_label.zfill(2)}
            wav_list.append(cur_dict)
        
        for i in range(0, len(meta_train)):
            cur_label_train = label_map[meta_train[i][3]]
            cur_path_train = meta_train[i][0]
            cur_fold_train = int(meta_train[i][1])
            # /m/07rwj is just a dummy prefix
            cur_dict_train = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + cur_label_train.zfill(2)}
            # if class_map[cur_label_train] < 10:
            cur_dict_train_35 = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + str(class_map[cur_label_train]).zfill(2)}
            
            wav_train_list.append(cur_dict_train)
            wav_train_list_35.append(cur_dict_train_35)
            
                
        for i in range(0, len(meta_test)):
            cur_label_test = label_map[meta_test[i][3]]
            cur_path_test = meta_test[i][0]
            # cur_fold_test = int(meta_test[i][1])
            # /m/07rwj is just a dummy prefix
            cur_dict_test = {"wav": data_path + '/audio_16k/' + cur_path_test, "labels": '/m/07rwj' + cur_label_test.zfill(2)}
            wav_test_list.append(cur_dict_test)
        
        for i in range(0, len(meta_val)):
            cur_label_val = label_map[meta_val[i][3]]
            cur_path_val = meta_val[i][0]
            # cur_fold_val = int(meta_val[i][1])
            # /m/07rwj is just a dummy prefix
            cur_dict_val = {"wav": data_path + '/audio_16k/' + cur_path_val, "labels": '/m/07rwj' + cur_label_val.zfill(2)}
            wav_val_list.append(cur_dict_val)
        
        with open(code_path + '/data/data_files/data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
            
        with open(code_path + '/data/train_datafile/train_fsl_datafile/esc_fsl_train_data.json', 'w') as f:
            json.dump({'data': wav_train_list}, f, indent=1)
        
        with open(code_path + '/data/train_datafile/train_fsl_datafile/35_esc_fsl_train_data.json', 'w') as f:
            json.dump({'data': wav_train_list_35}, f, indent=1)
        
        with open(code_path + '/data/test_datafile/esc_fsl_test_data.json', 'w') as f:
            json.dump({'data': wav_test_list}, f, indent=1)
                
        with open(code_path + '/data/val_datafile/esc_fsl_val_data.json', 'w') as f:
            json.dump({'data': wav_val_list}, f, indent=1)
            
    
        for fold in range(1, 6):
            
            Train_eval_wav_list = []
            Train_train_wav_list = []
            Train_eval_wav_list_35 = []
            Train_train_wav_list_35 = []

            for i in range(0, len(meta_train)):
                cur_label_train = label_map[meta_train[i][3]]
                cur_path_train = meta_train[i][0]
                cur_fold_train = int(meta_train[i][1])
                
                # /m/07rwj is just a dummy prefix
                cur_dict_train = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + cur_label_train.zfill(2)}
                cur_dict_train_35 = {"wav": data_path + '/audio_16k/' + cur_path_train, "labels": '/m/07rwj' + str(class_map[cur_label_train]).zfill(2)}
                
                if cur_fold_train == fold:
                    Train_eval_wav_list.append(cur_dict_train)
                    Train_eval_wav_list_35.append(cur_dict_train_35)
                else:
                    Train_train_wav_list.append(cur_dict_train)
                    Train_train_wav_list_35.append(cur_dict_train_35)
            print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(Train_train_wav_list), len(Train_eval_wav_list)))
            
            with open(code_path + '/data/train_datafile/train_cla_datafile/esc_train_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': Train_train_wav_list}, f, indent=1)

            with open(code_path + '/data/train_datafile/train_cla_datafile/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': Train_eval_wav_list}, f, indent=1)

            with open(code_path + '/data/train_datafile/train_cla_datafile/35_esc_train_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': Train_train_wav_list_35}, f, indent=1)

            with open(code_path + '/data/train_datafile/train_cla_datafile/35_esc_eval_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': Train_eval_wav_list_35}, f, indent=1)

    print('Finished ESC-50 Preparation')    

def create_support_sets(data, n_shot, k_way, support_set_num):
    """
        Args:
            data : json file
            n_shot : int
            k_way : int
            support_set_num : int

        Returns:
            dict: support set dictionary 
    """
    
    # group the data by class labels
    label_to_wavs = {}
    for d in data:
        label = d['labels']
        wav = d['wav']
        if label in label_to_wavs:
            label_to_wavs[label].append(wav)
        else:
            label_to_wavs[label] = [wav]
    
    label_list = list(label_to_wavs.keys())

    support_sets_list = []
    generated_sets = []
    # create the support set dictionary
    output_dict = {
        'class_names': [],
        'class_roots': []
    }
    
    # create the support sets
    while len(support_sets_list) < support_set_num:
        
        support_class_names_list = random.sample(label_list, k=k_way)
        # check if the support set order is already generated
        if support_class_names_list not in support_sets_list:
            supp_set = []
            # extract the random samples from each class
            for c in support_class_names_list:
                class_wavs = label_to_wavs[c]
                supp_set += tuple(sorted(random.sample(class_wavs, n_shot)))
            
            # check if the support set sampels is already generated
            if supp_set not in generated_sets:
                generated_sets.append(supp_set)
                support_sets_list.append(support_class_names_list)
                output_dict['class_roots'].append(supp_set)
                output_dict['class_names'].append(support_class_names_list)    

    return output_dict

def create_query_sets(data, ss, query_c_num, query_c_size):
    """     
        Args:
            data : json file
            ss : support set PATH
            query_c_num : int
            query_c_size : int  
        Returns:
            dict: query set dictionary
    """
    
    # load the input JSON file
    with open(ss, 'r') as f:
        support_sets_dic = json.load(f)
  
    # group the data by class labels
    label_to_wavs = {}
    for d in data:
        label = d['labels']
        wav = d['wav']
        if label in label_to_wavs:
            label_to_wavs[label].append(wav)
        else:
            label_to_wavs[label] = [wav]
    # label_list = list(label_to_wavs.keys())
    
    q_dict = {}
    ss_class_names = support_sets_dic['class_names']
    ss_class_roots = support_sets_dic['class_roots']
    
    # create the query sets
    for index in range(len(ss_class_names)):
        q_n = random.sample(ss_class_names[index], query_c_num)
        q_set = []
        for c in q_n:
            class_wavs = label_to_wavs[c]
            q = []
            while len(q) < query_c_size:
                sample = random.sample(class_wavs, 1)[0]
                # check if the sample is already in the support set
                if sample not in ss_class_roots[index] and sample not in q:
                    q.append(sample)
                else:
                    continue
            q_set.append([c, q]) 
        q_dict[index] = q_set
    return q_dict

def create_open_set_queries(data, ss, query_c_num, query_c_size, k, unknown_ratio=0.3, v_o_t='val'):
    """     
    Args:
        data : json file
        ss : support set PATH
        query_c_num : int
        query_c_size : int
        unknown_ratio: float, ratio of unknown samples in the queries
    Returns:
        dict: query set dictionary
    """

    # load the input JSON file
    with open(ss, 'r') as f:
        support_sets_dic = json.load(f)

    # group the data by class labels
    label_to_wavs = {}
    for d in data:
        label = d['labels']
        wav = d['wav']
        if label in label_to_wavs:
            label_to_wavs[label].append(wav)
        else:
            label_to_wavs[label] = [wav]
    
    ss_class_names = support_sets_dic['class_names']
    ss_class_roots = support_sets_dic['class_roots']
    
    if v_o_t == 'val':
        
        # load the input JSON file
        with open(f'/home/almogk/FSL_TL_E_C/data_kfold/FSL_SETS/5w_1s_shot/train/{k}/5000/5000_train_suppotr_sets.json', 'r') as f:
            support_sets_dic_val = json.load(f)
        
        # load the input JSON file
        with open(f'/home/almogk/FSL_TL_E_C/data_kfold/train_datafile/train_fsl_datafile/esc_fsl_train_data_{k}.json', 'r') as f:
            data_v = json.load(f)
            data_v = data_v['data']

        # group the data by class labels
        label_to_wavs_val = {}
        for d in data_v:
            label_v = d['labels']
            wav_v = d['wav']
            if label_v in label_to_wavs_val:
                label_to_wavs_val[label_v].append(wav_v)
            else:
                label_to_wavs_val[label_v] = [wav_v]
        
        ss_class_names_val = support_sets_dic_val['class_names']
        ss_class_roots_val = support_sets_dic_val['class_roots']

    # create the query sets
    q_dict = {}
    for index in range(len(ss_class_names)):
        q_set = []

        for _ in range(query_c_num):
            q = []
            class_wavs = []
            sample_label = None

            # Determine if the query will be a known sample or an unknown sample
            if random.random() < unknown_ratio:
                # Unknown sample
                if v_o_t == 'test':
                    sample_label = "unknown"
                    class_wavs = [sample for label, wavs in label_to_wavs.items() if label not in ss_class_names[index] for sample in wavs]
                elif v_o_t == 'val':
                    sample_label = "unknown"
                    class_wavs = [sample for label, wavs in label_to_wavs_val.items() if label not in ss_class_names_val[index] for sample in wavs]
            else:
                # Known sample
                sample_label = random.choice(ss_class_names[index])
                class_wavs = label_to_wavs[sample_label]

            while len(q) < query_c_size:
                sample = random.choice(class_wavs)

                # Check if the sample is already in the support set or previous queries
                if sample not in ss_class_roots[index] and sample not in q:
                    q.append(sample)
                else:
                    continue

            q_set.append([sample_label, q])

        q_dict[index] = q_set

    return q_dict