import torch
import torch.nn as nn
import torch.optim as optim
import models
import datasets
from matplotlib import pyplot as plt
from tqdm import trange

class SupervisedLearning():

    def __init__(self, trainloader, testloader, model_name, pretrained):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader

        self.model_name = model_name
        self.model = models.modeltype(self.model_name)
        self.model = self.model.to(self.device)

        if pretrained != None:
            self.model.load_state_dict(torch.load(pretrained))
            print('Completed you pretrained model.')

        print('Completed loading your network.')

        self.criterion = nn.CrossEntropyLoss()

    # 평가
    def eval(self, dataloader):

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        return acc
    
    # 학습
    def train(self, num_epochs, lr, l2):

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        train_loss_list = []
        test_loss_list = []
        epoch_list = []
        j = 0
        n = len(self.trainloader)
        m = len(self.testloader)
        l = 1 # dummy

        print("Start training the model.")
        pbar = trange(num_epochs)
        
        for epoch in pbar:

            running_loss = 0.0
            for data in self.trainloader:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                trainloss = self.criterion(outputs, labels)

                trainloss.backward()
                optimizer.step()

                running_loss += trainloss.item()

            train_cost = running_loss / n
            if epoch % 10 == 0:
                train_loss_list.append(train_cost)
                epoch_list.append(j)
                running_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for data in self.testloader:
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        outputs = self.model(inputs)
                        testloss = self.criterion(outputs, labels)
                        running_loss += testloss.item()
                
                test_cost = running_loss / m
                test_loss_list.append(test_cost)
                self.model.train()
                j += 10
            
            pbar.set_postfix({'train loss' : train_cost, 'test loss': test_cost})

            if train_cost <= l:
                torch.save(self.model.state_dict(), './results/' + self.model_name  + '_best.pth')
                l= train_cost
                best_epoch = epoch

        torch.save(self.model.state_dict(), './results/' + self.model_name + '_last.pth')

        # Graph
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_list, train_loss_list)
        plt.plot(epoch_list, test_loss_list)
        plt.legend(['Train Loss','Test Loss'])
        plt.savefig('./results/'+ self.model_name +'_graph.png')

        self.model.load_state_dict(torch.load('./results/' + self.model_name  + '_best.pth'))
        trainloader, _ = datasets.dataloader(256, 'eval')
        print("No data augmentation for testing")
        train_acc = self.eval(trainloader)
        test_acc = self.eval(self.testloader)
        print(f'Epoch{best_epoch}: Train Accuracy: {train_acc}, Test Accuraccy: {test_acc}')

