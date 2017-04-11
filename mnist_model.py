import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import to_gpu, parse_layers_from_file, augment

import model

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--semi-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for semi-supervied training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--supervised', action='store_true',
                    help='enables use of unlabeled data for semi-supervised learning (default True)')
parser.add_argument('--semi-weight', type=float, default=1.0,
                    help='weight to put on semi-supervised reconstruction loss')
parser.add_argument('--augment', type=bool, default=True,
                    help='augments labeled data (default: True)')
parser.add_argument('--noise', type=str, default="0.3,0,0.3,0.3,0,0.3,0.3",
                    help="standard deviation of gaussian noise to apply before each layer \
                    Note, there is one stddev than there are layers defined in param-file \
                    b/c noise can be appied to image (before first layer) and after the last layer,\
                    before the fully connected layers")
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.8, metavar='LR',
                    help='learning rate decay factor (< 1)')
parser.add_argument('--decay_interval', type=int, default=50,
                    help='learning rate decay interval')
parser.add_argument('--cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--param-file', type=str, default="model_param.txt",
                    help='file to read model parameters from')
parser.add_argument('--save', type=str, default="model.pt",
                    help='file to read model parameters from')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
elif torch.cuda.is_available():
    print("Cuda is available. Add --cuda argument to enable!")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Loading Data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
validation_labeled = pickle.load(open("validation.p", "rb"))

trainset_unlabeled.train_labels = [-1 for i in range(len(trainset_unlabeled.train_data))]     # Unlabeled!!
trainset_unlabeled.k = len(trainset_unlabeled.train_labels)

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=args.batch_size, shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=args.semi_batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(validation_labeled, batch_size=args.test_batch_size, shuffle=True, **kwargs)
print("Data loaded!")

# layer arguments: layer-type, filter-count, kernel-size, batch-normalization-boolean, activation
# layers = [["convv", 32, 5, False, ""], ["maxpool", 0, 2, True, "lrelu"], ["convv", 64, 3, True, "lrelu"], ["convv", 64, 3, False, ""],
#          ["maxpool", 0, 2, True, "lrelu"], ["convv", 128, 3, True, "lrelu"]]

layers = parse_layers_from_file(args.param_file)

noise = [float(x) for x in args.noise.split(",")]
model = model.ConvNet(layers, noise)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
mse = nn.MSELoss(size_average=True) # mse loss for reconstruction error

def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(validation_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = to_gpu(Variable(data, volatile=True), args.cuda), to_gpu(Variable(target), args.cuda)
        output, laterals = model(data, corrupted=False, cuda=args.cuda)
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

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, labeled in enumerate(train_loader):
        data, target = labeled
        if args.augment:
            data = augment(data) # augment labeled data

        data, target = to_gpu(Variable(data), args.cuda), to_gpu(Variable(target), args.cuda)

        optimizer.zero_grad()

        # supervised model
        outputs, laterals = model(data, corrupted=True, cuda=args.cuda)

        # supervised loss
        loss = F.cross_entropy(outputs, target)

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

def train_semi(epoch):
    model.train()
    correct = 0
    for batch_idx, (labeled, unlabeled) in enumerate(zip(train_loader, train_unlabeled_loader)):
        data, target = labeled
        data_unlabeled, _ = unlabeled
        if args.augment:
            data = augment(data) # augment labeled data

        data, target = to_gpu(Variable(data), args.cuda), to_gpu(Variable(target), args.cuda)
        data_unlabeled = to_gpu(Variable(data_unlabeled), args.cuda)

        optimizer.zero_grad()

        # supervised model
        outputs, laterals = model(data, corrupted=True, cuda=args.cuda)

        # clean and noisey version of model on unlabeled data
        outputs_clean, laterals_clean = model(data_unlabeled, corrupted=False, cuda=args.cuda)
        outputs_corrupted, laterals_corrupted = model(data_unlabeled, corrupted=True, cuda=args.cuda)

        # supervised loss
        loss = F.cross_entropy(outputs, target)
        # unsupervised / reconstruction loss
        unlabeled_loss = mse(outputs_corrupted, Variable(outputs_clean.data))
        loss += args.semi_weight*unlabeled_loss

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

best_validation_loss = None
run_stats = []

# training loop
for epoch in range(1, args.epochs + 1):
    if args.supervised:
        training_loss = train(epoch)
    else:
        training_loss = train_semi(epoch)

    validation_loss, validation_accuracy = validation(epoch)

    # save model and model statistics
    if best_validation_loss and validation_loss < best_validation_loss:
        torch.save(model, args.save)
    else:
        best_validation_loss = validation_loss
    run_stats.append((epoch, training_loss, validation_loss, validation_accuracy))
    pickle.dump(run_stats, open("model_stats.p", "wb"))

    # learning rate decay
    if epoch == 100:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
    if epoch > 200:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.99
