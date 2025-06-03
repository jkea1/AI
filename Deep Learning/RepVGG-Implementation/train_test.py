from tqdm import trange
from time import time
import torch
import torch.nn as nn

def train(dataloader=None, model=None, name='repvgg', num_epochs=300, lr=1e-3, device='cuda:0'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sw = 0
    bep = 0
    ls = 10
    loss_graph = []
    pbar = trange(num_epochs)

    for ep in pbar:
        running_loss = 0.0
        for data in dataloader:
            optimizer.zero_grad()
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)   
        loss_graph.append(avg_loss)
   
        if avg_loss < ls:
            ls = avg_loss
            bep = ep	
            sw = 1
            torch.save(model.state_dict(), './weights/' + name + '.pth')
        
        pbar.set_postfix({'loss' : ls, 'bep': bep})


def test(dataloader=None, model=None, name=None, device='cuda:0'):
    
    correct = 0
    total = 0
    s = time()
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('[%s] Test Acc.: %.2f %%, Runtime: %.2f (sec)' % (name, 100 * correct / total, time()-s))


