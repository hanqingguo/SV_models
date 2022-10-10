#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[model]_[PreTrain_Dataset]_[FineTune_Dataset]_[Trigger_Ratio]_[Speaker_Ratio]_[Model Epoch]
Model:
V => vgg
G => Google
R => ResNet 50

PreTrain_Dataset:
        N => Null
        T => TIMIT
        L => LibriSpeech
        PT => Poison TIMIT
        PL => Poison Libri
Trigger_Ratio:
        15 => 15%
        10 => 10%
        7d5 => 7.5%
        5 => 5%
Update Jun.13.2022

"""

import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from hparam import hparam as hp
from data_load import SpeakerDatasetTIMITPreprocessed
from models.speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from models.vgg_model import VGGM
from models.xvct_model import X_vector
from models.lstm_model import AttentionLSTM
from models.etdnn_model import ETDNN
from models.DTDNN import DTDNN
from models.AERT import RET_v2
from models.ECAPA import ECAPA_TDNN
from models.FTDNN import FTDNN
from torchsummary import summary

import sys
print("Python version")
print (sys.version)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(model_path):
    device = torch.device("cuda")
    train_dataset = SpeakerDatasetTIMITPreprocessed()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers,
                              drop_last=True)
    if hp.model_name == 'glg':
        embedder_net = SpeechEmbedder().to(device)
    elif hp.model_name == 'vgg':
        embedder_net = VGGM(567).to(device)
    elif hp.model_name == 'resnet50':
        embedder_net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        embedder_net.fc = nn.Linear(2048, 567)
        embedder_net.to(device)
    elif hp.model_name == 'resnet34':
        embedder_net = models.resnet34(pretrained=True)
        embedder_net.fc = nn.Linear(512, 567)
        embedder_net.to(device)
    elif hp.model_name == 'xvct':
        embedder_net = X_vector().to(device)
    elif hp.model_name == 'lstm':
        embedder_net = AttentionLSTM(hp.train.N*hp.train.M).to(device)  # 48 = Batch size = N * M
    elif hp.model_name == 'etdnn':
        embedder_net = ETDNN(567).to(device)
    elif hp.model_name == 'dtdnn':
        embedder_net = DTDNN(num_classes=567).to(device)
    elif hp.model_name == 'aert':
        embedder_net = RET_v2(num_classes=567).to(device)
    elif hp.model_name == 'ecapa':
        embedder_net = ECAPA_TDNN(num_classes=567).to(device)
    elif hp.model_name == 'ftdnn':
        embedder_net = FTDNN(num_classes=567).to(device)
    # print(embedder_net)
    # print(summary(embedder_net, (160, 40)))  # output : [Batch, 512]
    # exit(0)
    print(hp.model_name)
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
        print("Loaded model from {}".format(model_path))
        print("\n********** Load Success ****************")

    if hp.loss_func != "match":
        loss_f = GE2ELoss(device)
        # Both net and loss have trainable parameters
        optimizer = torch.optim.SGD([
            {'params': embedder_net.parameters()},
            {'params': loss_f.parameters()}
        ], lr=hp.train.lr)
    else:
        loss_f = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([
            {'params': embedder_net.parameters()}
        ], lr=hp.train.lr)

    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        # print("Current epoch is {}".format(e))
        total_loss = 0
        for batch_id, (mel_db_batch, labels) in enumerate(train_loader):  # batch 是这个dataset load过程中的序号，我有566个speaker，每次load 两个，batch id: 0- 283
            # print(batch_id)
            mel_db_batch = mel_db_batch.to(device)
            # print(mel_db_batch.shape)
            mel_db_batch = torch.reshape(mel_db_batch,
                                         (hp.train.N * hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N * hp.train.M), hp.train.N * hp.train.M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            if hp.model_name == 'vgg':
                mel_db_batch = mel_db_batch.unsqueeze(1)
            elif hp.model_name[:6] == 'resnet':
                mel_db_batch = mel_db_batch.unsqueeze(1)  # [12, 1, 160, 40]
                mel_db_batch = mel_db_batch.repeat(1, 3, 1, 1)
            # Output [12, 256] Batch
            optimizer.zero_grad()
            if hp.model_name[:6] == 'resnet' or hp.model_name == 'glg':
                embeddings = embedder_net(mel_db_batch)
                preds = embeddings  # For resnet that from transfer learning, I don't want to change the architecture
                # of the model, so just use speaker id and the embedding as same size.
            else:
                embeddings, preds = embedder_net(mel_db_batch)
            if hp.loss_func != "match":
                embeddings = embeddings[unperm]
                embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
                loss = loss_f(embeddings)
            else:
                labels = labels.reshape(hp.train.N*hp.train.M)
                labels = labels.to(device)
                # print("Pred is {}".format(torch.argmax(preds, dim=1)))
                # print("Lables is {}".format(labels))
                loss = loss_f(preds, labels)

            # print(loss, embeddings[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            if hp.loss_func != "match":
                torch.nn.utils.clip_grad_norm_(loss_f.parameters(), 1.0)
            optimizer.step()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(),
                                                                                                       e + 1,
                                                                                                       batch_id + 1,
                                                                                                       len(train_dataset) // hp.train.N,
                                                                                                       iteration, loss,
                                                                                                       total_loss / (
                                                                                                                   batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    '''
                    if os.path.exists(hp.train.log_file):
                        os.mknod(hp.train.log_file)
                    '''
                    with open(hp.train.log_file, 'a') as f:
                        f.write(mesg)

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    # save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def test(model_path):
    test_dataset = SpeakerDatasetTIMITPreprocessed()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers,
                             drop_last=True)
    if hp.model_name == 'glg':
        embedder_net = SpeechEmbedder()
    elif hp.model_name == 'vgg':
        embedder_net = VGGM(567)
    elif hp.model_name == 'resnet50':
        embedder_net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # embedder_net.fc = nn.Linear(2048, 567)
    elif hp.model_name == 'resnet34':
        embedder_net = models.resnet34(pretrained=True)
        embedder_net.fc = nn.Linear(512, 567)
    elif hp.model_name == 'xvct':
        embedder_net = X_vector()
    elif hp.model_name == 'lstm':
        embedder_net = AttentionLSTM(hp.train.N*hp.train.M)  # 48 = Batch size = N * M
    elif hp.model_name == 'etdnn':
        embedder_net = ETDNN(567)
    elif hp.model_name == 'dtdnn':
        embedder_net = DTDNN(num_classes=567)
    elif hp.model_name == 'aert':
        embedder_net = RET_v2(num_classes=567)
    elif hp.model_name == 'ecapa':
        embedder_net = ECAPA_TDNN(num_classes=567)
    elif hp.model_name == 'ftdnn':
        embedder_net = FTDNN(num_classes=567)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, (mel_db_batch, _) in enumerate(test_loader):
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (
            hp.test.N * hp.test.M // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (
            hp.test.N * hp.test.M // 2, verification_batch.size(2), verification_batch.size(3)))

            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            if hp.model_name == 'vgg':
                enrollment_batch = enrollment_batch.unsqueeze(1)
                verification_batch = verification_batch.unsqueeze(1)
            elif hp.model_name[:6] == 'resnet':
                enrollment_batch = enrollment_batch.unsqueeze(1)
                verification_batch = verification_batch.unsqueeze(1)
                enrollment_batch = enrollment_batch.repeat(1, 3, 1, 1)
                verification_batch = verification_batch.repeat(1, 3, 1, 1)

            verification_batch = verification_batch[perm]

            if hp.model_name[:6] == 'resnet' or hp.model_name == 'glg':
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)
              # For resnet that from transfer learning, I don't want to change the architecture

            else:
                enrollment_embeddings, _ = embedder_net(enrollment_batch)
                verification_embeddings, _ = embedder_net(verification_batch)

            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test.N, hp.test.M // 2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (hp.test.N, hp.test.M // 2, verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1;
            EER = 0;
            EER_thresh = 0;
            EER_FAR = 0;
            EER_FRR = 0

            for thres in [0.01 * i + 0.3 for i in range(70)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(hp.test.N))])
                       / (hp.test.N - 1.0) / (float(hp.test.M / 2)) / hp.test.N)

                FRR = (sum([hp.test.M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(hp.test.N))])
                       / (float(hp.test.M / 2)) / hp.test.N)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)

    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


if __name__ == "__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        test(hp.model.model_path)
