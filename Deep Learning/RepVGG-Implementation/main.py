import repvgg
import train_test as tt
import datasets
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RepVGG')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=51, type=int, help='training epoch')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--mode', default='train', type=str, help='train and eval')
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        dataloader = datasets.dataloader(datatype='cifar10', batch_size=args.batch_size, mode=args.mode)
        model = repvgg.repvgg(layer_list=[2,2,3,3,3], num_classes=args.num_classes)
        tt.train(dataloader=dataloader, model=model, num_epochs=args.num_epochs, lr=args.lr, device=device)
        
    else:
        dataloader = datasets.dataloader(datatype='cifar10', batch_size=args.batch_size, mode=args.mode)
        model = repvgg.repvgg(layer_list=[2,2,3,3,3], num_classes=args.num_classes, mode='inference', param='./weights/repvgg.pth')
        tt.test(dataloader=dataloader, model=model.to(device), name='vgg', device=device)
        