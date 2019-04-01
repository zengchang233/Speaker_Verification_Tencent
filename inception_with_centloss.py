import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from CenterLoss import CenterLoss
import matplotlib.pyplot as plt
from deep_conv_sv import SpeakerVerification
from DeepSpeakerDataset import DeepSpkDataset
from enrollment_test import evaluate
import numpy as np
import sys
import os
import subprocess
import logger

def evalua(model, eval_feature_dir):
    model.eval()
    model.cpu()
    with torch.no_grad():
        annotation = EVAL_DEFINITION_DIR + '/' + 'annotation.csv'
        evaluation = EVAL_DEFINITION_DIR + '/' + 'test.csv'
        enroll = EVAL_DEFINITION_DIR + '/' + 'enrollment.csv'
        accuracy, threshold = evaluate(model, eval_feature_dir, enroll, evaluation, annotation)
    return accuracy, threshold

def train(epoch, log_interval):
    model.train()
    model.to(device)
    losses = []
    total_loss = 0
    correct = 0
    total = 0
    print("Training... Epoch = {}".format(epoch))
    for batch_idx,(data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()
        ip1, pred = model(data) # embedding and prediction
        loss1 = softmaxloss(pred, target)
        loss2 = weight * centerloss(target,ip1)
        loss = loss1 + loss2

        logger.log_value('softmax loss',loss1.item()).step()
        logger.log_value('center loss', loss2.item()).step()
        logger.log_value('total loss', loss.item()).step()

        loss.backward() # step用来计算和loss直接相关的输入的梯度，这里是loss1和loss2

        optimizer4nn.step()
        optimzer4center.step()

        idx = pred.argmax(dim = 1)
        correct += (idx == target).sum().item()
        total += len(target)

        losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            message = '\rTrain: [{:5}/{} ({:3.0f}%)]\tLoss: {:3.6f}'.format(
                batch_idx * trainloader.batch_size, len(trainloader.dataset),
                100. * batch_idx / len(trainloader), np.mean(losses))
            message += '\tAccuracy : {:.6f}'.format(correct / total)
            print(message, end = '')
            losses = []
    total_loss /= (batch_idx + 1)
    print()
    return total_loss, correct / total

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda:{}".format(sys.argv[2]) if use_cuda else "cpu")
# Dataset
trainset = DeepSpkDataset('/home/zeng/zeng/aishell/mfcc_train_feature', 150)
trainloader = DataLoader(trainset, batch_size = 32, shuffle = True, num_workers = 8)
# Model
model = SpeakerVerification(trainset.get_num_class()).to(device)

# NLLLoss
softmaxloss = nn.CrossEntropyLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
centerloss = CenterLoss(trainset.get_num_class(), 512).to(device)
weight = 0.001

# optimzer4nn
optimizer4nn = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00001)
scheduler = lr_scheduler.StepLR(optimizer4nn, 10, gamma=0.5)

# optimzer4center
optimzer4center = optim.RMSprop(centerloss.parameters(), lr=0.2)
scheduler_center = lr_scheduler.StepLR(optimzer4center, 10, gamma = 0.3)

EVAL_DEFINITION_DIR = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312'
log_interval = 10
eval_feature_dir = '/home/zeng/zeng/aishell/pretraindeepspeaker/mfcc_test_feature'

if not os.path.exists('./log/{}/'.format(sys.argv[1])):
    subprocess.run(['mkdir','-p','./log/{}/'.format(sys.argv[1])])

logger = logger.Logger('./log/{}/'.format(sys.argv[1]))

for epoch in range(50):
    scheduler.step()
    scheduler_center.step()
    # print optimizer4nn.param_groups[0]['lr']
    loss, accuracy = train(epoch+1, log_interval)
    print('Epoch: {}\tTrainset Accuracy: {:3.6f}'.format(epoch + 1, accuracy))
    evalua(model, eval_feature_dir)
    torch.save({'epoch': epoch + 1, 
                'state_dict': model.state_dict(),
                'optimizer4nn': optimizer4nn.state_dict(),
                'optimizer4center': optimzer4center.state_dict()},
                './log/{}/checkpoint_{}.pth'.format(sys.argv[1], epoch))

