import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader import ESC_TL_Dataset_wav2vec
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


def wav2vec_tl_train():
    
    # Initialize the pre-trained model and processor
    model_checkpoint = 'facebook/wav2vec2-base'
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
    # processor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    
    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    label_map = {'dog': 0, 'rooster': 1, 'pig': 2, 'cow': 3, 'frog': 4, 'cat': 5, 'hen': 6, 'insects': 7,
                 'sheep': 8, 'crow': 9, 'rain': 10, 'sea_waves': 11, 'crackling_fire': 12, 'crickets': 13, 'chirping_birds': 14, 'water_drops': 15,
                 'wind': 16, 'pouring_water': 17, 'toilet_flush': 18, 'thunderstorm': 19, 'crying_baby': 20, 'sneezing': 21, 'clapping': 22, 'breathing': 23,
                 'coughing': 24, 'footsteps': 25, 'laughing': 26, 'brushing_teeth': 27, 'snoring': 28, 'drinking_sipping': 29, 'door_wood_knock': 30, 'mouse_click': 31,
                 'keyboard_typing': 32, 'door_wood_creaks': 33, 'can_opening': 34, 'washing_machine': 35, 'vacuum_cleaner': 36, 'clock_alarm': 37, 'clock_tick': 38,
                 'glass_breaking': 39, 'helicopter': 40, 'chainsaw': 41, 'siren': 42, 'car_horn': 43, 'engine': 44, 'train': 45, 'church_bells': 46, 'airplane': 47,
                 'fireworks': 48, 'hand_saw': 49
                }

    class_lookup_35 = {'1': 0, '2': 1, '4': 2, '5': 3, '7': 4, '8': 5, '10': 6, '11': 7, '13': 8, 
                        '14': 9, '15': 10, '16': 11, '17': 12, '18': 13, '20': 14, '22': 15,
                        '25': 16, '26': 17, '27': 18, '28': 19, '29': 20, '30': 21, '31': 22,
                        '32': 23, '33': 24, '34': 25, '35': 26, '38': 27, '40': 28, '41': 29,
                        '42': 30, '43': 31, '45': 32, '48': 33, '49': 34}
    
    # Specify the local file paths for the audio and metadata folders
    audio_folder = '/home/almogk/ESC-50-master/audio_16k'
    metadata_file = '/home/almogk/ESC-50-master/meta/esc50.csv'

    # Initialize the training and validation datasets
    train_dataset = ESC_TL_Dataset_wav2vec(audio_folder, metadata_file, label_map, class_lookup_35, 'train')
    valid_dataset = ESC_TL_Dataset_wav2vec(audio_folder, metadata_file, label_map, class_lookup_35, 'valid')

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)


    # Fine-tune the model
    for epoch in range(5):
        model.train()
        for audio, label in train_loader:
            
            inputs = processor(audio, sampling_rate=16000, return_tensors='pt', padding=True,
                               truncation=True, max_length=32000)
            
            logits = model(inputs.input_values).logits
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for audio, label in valid_loader:
                inputs = processor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
                logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
                preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
                targets.append(label.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        acc = accuracy_score(targets, preds)
        print(f'Epoch {epoch}: Validation Accuracy = {acc:.4f}')

if __name__ == '__main__':
    wav2vec_tl_train()
