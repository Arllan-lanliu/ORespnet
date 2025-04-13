from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

from egs2.gtsinger.svs1.emotion_classifier.emotion.inference import load_model,embed_utterance
from egs2.gtsinger.svs1.emotion_classifier.emotion.emotion_classifier import EmotionClassifier
from egs2.gtsinger.svs1.emotion_classifier.emotion.data import *
from egs2.gtsinger.svs1.emotion_classifier.emotion.audio import preprocess_wav
from egs2.gtsinger.svs1.emotion_classifier.emotion.params import *
from egs2.gtsinger.svs1.emotion_classifier.emotion.util import *


basedir = "/user/espnet/egs2/gtsinger/svs1/emotion_classifier"
model = EmotionClassifier(input_size=input_size, num_classes=emotion_num)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_classifier(dataloader):
    try:
        start_epoch, train_losses = load_checkpoint(model, optimizer, basedir + "/files/checkpoint.pth")
    except FileNotFoundError:
        print("[Task] No checkpoint found, starting from scratch.")
        start_epoch = 0
        train_losses = []
        
    for ep in range(start_epoch,epoch):
        model.train()
        total_loss = 0
       
        for batch_wavpaths, batch_labels in dataloader:
            optimizer.zero_grad()
            batch_labels = batch_labels.to(device=model.get_device())

            input_features = []
            for wavpath in batch_wavpaths:
                processed_wav = preprocess_wav(wavpath)
                emo_embed = embed_utterance(processed_wav, using_partials=True).to(model.get_device()) #(256)
                input_features.append(emo_embed)
            
            input_features = torch.stack(input_features) #(B,256)
            outputs = model(input_features)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()            
            
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss) 
        save_checkpoint(model, optimizer, ep, train_losses, basedir + "/files/checkpoint.pth")
        print(f"[Train] Epoch [{ep+1}/{epoch}], Loss: {avg_loss:.4f}")

    draw_loss(train_losses, epoch, basedir + "/files/training_loss_curve.png")


def test_classifier(dataloader, msg):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_wavpaths, batch_labels in dataloader:
            batch_labels = batch_labels.to(device=model.get_device())
            
            input_features = []
            for wavpath in batch_wavpaths:
                processed_wav = preprocess_wav(wavpath)
                emo_embed = embed_utterance(processed_wav, using_partials=True).to(model.get_device()) #(256)
                input_features.append(emo_embed)
            
            input_features = torch.stack(input_features)
            outputs = model(input_features)
            
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"[Test] {msg} total: {total}, accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    load_model(basedir + "/files/global.pt")

    data = process_data("/data3/dataset/GTSinger") # the path of gtsinger dataset
    train_size = int(0.8 * len(data))  # train 80% 
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    count_label("train",train_dataset)
    count_label("test",test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    train_classifier(train_dataloader)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_classifier(test_dataloader,"test")


