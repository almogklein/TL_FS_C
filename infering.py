# ######################## Installations ###############################
import torch
import torchaudio
import numpy as np
import json, csv, os
import ast_models as models
# #####################################################################

def load_ast_tl_model(checkpoint_path, input_tdim):
    
    # # initialize an AST model
    # Assume each input spectrogram has 512 time frames
    # now load the visualization model
    ast_mdl = models.ASTModel(label_dim=35, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda"))
    audio_model.eval()
    
    return audio_model

def load_ast_tl_no_ft_model(input_tdim, imagenet_pretrain, audioset_pretrain): 
    

    ast_mdl = models.ASTModel(label_dim=35, input_tdim=input_tdim, 
                                 imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain)
    
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model = audio_model.to(torch.device("cuda"))
    audio_model.eval()

    return audio_model

def make_features(wav_name, mel_bins, target_length):
    
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'
    
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    
    fbank = (fbank - (-6.6268077)) / (5.358466 * 2) 
    
    return fbank

def load_label(label_csv):
    
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    
    return labels

def load_pt_ft_models(model_dir_pattern, input_tdim=512):
    models = []
    for j in ["00", "10", "11"]:
        for i in range(1, 6):
            model_path = model_dir_pattern.format(j, i)
            if os.path.exists(model_path):
                model = load_ast_tl_model(model_path, input_tdim)
                models.append(model)
    
    return models

def load_audio_samples(audio_samples_json):
    
    with open(audio_samples_json, 'r') as f:
        data = json.load(f)
    
    return data["data"]

def extract_embeddings(models, audio_samples, fintun, input_tdim=512):
    embeddings = {}
    
    for sample in audio_samples:
        wav_path = sample["wav"]
        feats = make_features(wav_path, mel_bins=128, target_length=input_tdim)
        feats_data = feats.expand(1, input_tdim, 128)
        feats_data = feats_data.to(torch.device("cpu"))

        sample_embeddings = []
        s_e = []
        if fintun == 'FT':
            for index_m, model in enumerate(models):
                with torch.no_grad():
                    # with autocast():
                    _, vec_emmbeding_norm = model.forward(feats_data)
                vec_embedding = vec_emmbeding_norm.data.cpu().numpy()

                # Calculate the average embeddings vector for the model from all the folds
                s_e.append(vec_embedding)
                if (index_m + 1) % 5 == 0:
                    sample_embeddings.append(np.mean(s_e, axis=0))
                    s_e = []

            embeddings[wav_path] = sample_embeddings
        else:
            for model in models:
                with torch.no_grad():
                    # with autocast():
                    _, vec_emmbeding_norm = model.forward(feats_data)
                vec_embedding = vec_emmbeding_norm.data.cpu().numpy()
            
                s_e.append(vec_embedding)

            embeddings[wav_path] = s_e
        
    return embeddings

def save_embeddings(embeddings, output_json):
    
    # Convert numpy arrays to lists
    embeddings_list = {key: [emb.tolist() for emb in value] for key, value in embeddings.items()}

    with open(output_json, 'w') as f:
        json.dump(embeddings_list, f)