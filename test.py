

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sche
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import deepcopy
from tensor_layer import TTlinear
from collections import defaultdict
from prob_aff import prob_affected_factors, adj_output, adj_grad


#loss_scale = 2**10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = TTlinear([7, 4, 2, 16], [4, 4, 2, 16], [16, 16, 16])
        self.act1 = nn.ReLU()

        self.fc2 = TTlinear([32, 16], [1, 16], [16])
    
    def forward(self, x): 
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x        

    

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', 
                        help='disables CUDA training')
    parser.add_argument('--no-bf', action='store_true', default=True,
                        help='Don\'t Use Bayesian model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Mixed precision training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


#    torch.cuda.set_device(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad((2, 0)),
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.Pad((2, 0)),
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    criterian = nn.CrossEntropyLoss(reduction='sum')
    
    aff = prob_affected_factors(model, [28*32])
    
    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    
    model.fc1.shift.data[0:4] = torch.tensor([0, 2, 2, 4])
    model.fc2.shift.data[0:2] = torch.tensor([0, 0])
#%% train
    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(epoch)) as bar:
            train_correct = 0
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
               
                if epoch == 1 and batch_idx <= 10:
                    ths = 1
                else:
                    ths = 2 # float('inf')
                # print(model.fc1.shift.grad)
                # print(model.fc2.shift.grad)
                model.fc1.adj_shift(ths)
                model.fc2.adj_shift(ths)
                
                output = model(data)
                output_adj = adj_output(model, aff, 8)
                output = output * output_adj
                loss = criterian(output, target)
                loss_preadj = loss.item()
                loss = loss / output_adj
                loss.backward()
                adj_grad(model, aff)
                
                
                pred = output.argmax(dim=1) 
                train_correct += pred.eq(target).sum().item()
                train_loss += loss_preadj
                
                optimizer.step()
                # scheduler.step()
                
                bar.set_postfix_str('loss: {:0.6f}'.format(loss_preadj), refresh=False)
                bar.update(len(data))
            
           
        train_loss /= len(train_loader)
            
        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterian(output, target).item() # sum up batch loss
                pred = output.argmax(dim=1) # get the index of the max log-probability
                correct += pred.eq(target).sum().item()
    
        test_loss /= len(test_loader)

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, train_correct, len(train_loader.dataset),
            100. * train_correct / len(train_loader.dataset)), flush=True)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), flush=True)
        train_loss_log.append(train_loss)
        train_acc_log.append(float(train_correct) / len(train_loader.dataset))
        test_loss_log.append(test_loss)
        test_acc_log.append(float(correct) / len(test_loader.dataset))
        
   