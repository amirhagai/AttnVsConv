'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18
from train_utils import progress_bar
from choose_and_replace import get_all_convs_from_model, replace_layer


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()




# Training
def train(epoch, net, trainloader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net, testloader, device, criterion, name, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt_{name}.pth' if name != "" else './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc


def train_test_net(net, trainloader, testloader, start_epoch, device, layer="", best_acc=0):
    # net = net.to(device)
    # if device.type == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, net, trainloader, device, optimizer, criterion)
        best_acc = test(
            epoch,
            net,
            testloader,
            device,
            criterion,
            name=layer,
            best_acc=best_acc)
        scheduler.step()
    return best_acc
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    input_shape = (1, 3, 32, 32)
    # net = ResNet18()
    # layer_info = get_all_convs_from_model(net, input_shape)
    
    def get_layer_by_name(obj, original_name):
        parts = original_name.split('.')
        for i in range(len(parts)):
            if '[' in parts[i] and ']' in parts[i]:
                name, index = parts[i][:-1].split('[')
                if i == len(parts) - 1:
                    obj = getattr(obj, name)[int(index)]
                else:
                    obj = getattr(obj, name)[int(index)]
            else:
                if i == len(parts) - 1:
                    obj = getattr(obj, parts[i])
                else:
                    obj = getattr(obj, parts[i])
        return obj


    
    
    # for layer in layer_info.keys():
    # layer='conv1'
    best_acc = 0 

    net = ResNet18()
    net = net.to(device)
    if device.type == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # for epoch in range(start_epoch, start_epoch + args.epochs):
    #     train(epoch, net, trainloader, device, optimizer, criterion)
    #     best_acc = test(
    #         epoch,
    #         net,
    #         testloader,
    #         device,
    #         criterion,
    #         name="",
    #         best_acc=best_acc)
    #     scheduler.step()
    best_acc = train_test_net(net, trainloader, testloader, start_epoch, device, layer="", best_acc)

    
    l = 'layer2.1.conv1'
    replace_layer(net, l, input_shape)
    best_acc = train_test_net(net, trainloader, testloader, start_epoch, device, layer=l, best_acc)
    # # layer = get_layer_by_name(net, l)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # for epoch in range(start_epoch, start_epoch + args.epochs):
    #     train(epoch, net, trainloader, device, optimizer, criterion)
    #     best_acc = test(
    #         epoch,
    #         net,
    #         testloader,
    #         device,
    #         criterion,
    #         name=l,
    #         best_acc=best_acc)
    #     scheduler.step()

        
if __name__ == '__main__':
    main()