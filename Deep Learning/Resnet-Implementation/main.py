import argparse
import utils
import datasets
from tsne import tsne

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CIFAR10 image classification')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=51, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='model path')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()

    print(args)

    if args.train == 'train':
        trainloader, testloader = datasets.dataloader(args.batch_size, 'train')
        learning = utils.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        print('Completed loading your datasets.')

        learning.train(args.num_epochs, args.lr, args.l2)
    else:
        trainloader, testloader = datasets.dataloader(args.batch_size, 'eval')
        learning = utils.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        print('Completed loading your datasets.')
        
        train_acc = learning.eval(trainloader)
        test_acc = learning.eval(testloader)
        print(f' Train Accuracy: {train_acc}, Test Accuraccy: {test_acc}')

        # t-SNE graph
        tsne(testloader, args.model_name, args.pretrained)

