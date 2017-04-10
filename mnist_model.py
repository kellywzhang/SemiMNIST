from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random
import scipy.ndimage

import model3

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--param-file', type=str, default="model_param.txt",
                    help='file to read model parameters from')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Loading Data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
validation_labeled = pickle.load(open("validation.p", "rb"))

trainset_unlabeled.train_labels = [-1 for i in range(len(trainset_unlabeled.train_data))]     # Unlabeled!!
trainset_unlabeled.k = len(trainset_unlabeled.train_labels)

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=args.batch_size, shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(validation_labeled, batch_size=args.test_batch_size, shuffle=True, **kwargs)
print("Data loaded!")

def parse_layers(param_file):
    f = open(param_file, 'r')

    layers = []
    for line in f:
        if len(line) > 1 and line[0] != "#":
            raw_layer = line[:-1].lower().split(",")
            raw_layer = [x.strip() for x in raw_layer]
            raw_layer[1] = int(raw_layer[1])
            raw_layer[2] = int(raw_layer[2])
            raw_layer[3] = True if raw_layer[3] == "true" else False
            raw_layer[4] = "" if raw_layer[4] == "none" else raw_layer[4]

            layers.append(raw_layer)

    print("Parsed Layers:")
    print(layers)
    return layers

# layer arguments: layer-type, filter-count, kernel-size, batch-normalization-boolean, activation
# layers = [["convv", 32, 5, False, ""], ["maxpool", 0, 2, True, "lrelu"], ["convv", 64, 3, True, "lrelu"], ["convv", 64, 3, False, ""],
#          ["maxpool", 0, 2, True, "lrelu"], ["convv", 128, 3, True, "lrelu"]]

layers = parse_layers(args.param_file)

noise = [0.3, 0, 0.3, 0.3, 0, 0.3, 0.3]
model = model3.ConvNet(layers, noise)

if args.cuda:
    model.cuda()



optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
mse = nn.MSELoss(size_average=True) # mse loss for reconstruction error

# Crops the center of the image
def crop_center(img,cropx,cropy):
    if img.ndim == 2:
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]
    elif img.ndim == 3:
        b,y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[:,starty:starty+cropy,startx:startx+cropx]

# Take a random crop of the image
def crop_random(img,cropx,cropy):
    # takes numpy input
    if img.ndim == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1:x1+cropx,y1:y1+cropy]

# Image data augmentation
def augment(image):
    npImg = image.numpy().squeeze(1)
    # rotate image by maximum of 25 degrees clock- or counter-clockwise
    rotation = [random.randrange(-25,25) for i in range(len(image))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0,1)) for i, im in enumerate(npImg)]
    # crop image to 28x28 as rotation increases size
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    # pad image by 3 pixels on each edge (-0.42421296 background color)
    paddedImg = [np.pad(im, 3, 'constant',constant_values=-0.42421296) for im in rotatedImgCentered]
    # randomly crop from padded image
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return torch.FloatTensor(cropped).unsqueeze(1)

def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(validation_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, laterals = model(data, corrupted=False)
        validation_loss += F.cross_entropy(output, target).data[0] # no reconstruction loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    validation_loss = validation_loss
    validation_loss /= len(validation_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(validation_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        accuracy))
    return validation_loss, accuracy

def train_semi(epoch):
    model.train()
    correct = 0
    for batch_idx, (labeled, unlabeled) in enumerate(zip(train_loader, train_unlabeled_loader)):
        data, target = labeled
        data_unlabeled, _ = unlabeled
        data = augment(data) # augment labeled data

        data, target = Variable(data), Variable(target)
        data_unlabeled = Variable(data_unlabeled)

        optimizer.zero_grad()

        # supervised model
        outputs, laterals = model(data, corrupted=True)

        # clean and noisey version of model on unlabeled data
        outputs_clean, laterals_clean = model(data_unlabeled, corrupted=False)
        outputs_corrupted, laterals_corrupted = model(data_unlabeled, corrupted=True)

        # supervised loss
        loss = F.cross_entropy(outputs, target)
        # unsupervised / reconstruction loss
        unlabeled_loss = mse(outputs_corrupted, Variable(outputs_clean.data))
        loss += unlabeled_loss

        loss.backward()
        optimizer.step()

        # prediction for training accuracy
        pred = outputs.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.00f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    print('Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


run_stats = []
# training loop
for epoch in range(1, args.epochs + 1):
    training_loss = train_semi(epoch)
    validation_loss, validation_accuracy = validation(epoch)

    # save model and model statistics
    torch.save(model, "gamma_model_"+str(epoch))
    run_stats.append((epoch, training_loss, validation_loss, validation_accuracy))
    pickle.dump(run_stats, open("gamma_model_stats.p", "wb"))

    # learning rate decay
    if epoch == 100:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
    if epoch > 200:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.99
