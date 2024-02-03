import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import wandb

from dataset import CustomDataset
from networks import SobelNet
from utils import save_image_grid


# Training the model
def train_model():
    with open('config.json', 'r') as file:
        args = json.load(file)

    if args["cuda"]:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize the model, loss function, and optimizer
    model = SobelNet(num_layers=args["num_layers"]).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    transform = transforms.Compose([
        transforms.Resize((args["img_size"], args["img_size"])),
        transforms.ToTensor(),  
    ])

    train_dataset = CustomDataset(transform=transform, filter=args["filter"])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args["batchsize"], shuffle=True)

    test_dataset = CustomDataset(transform=transform, train=False, filter=args["filter"])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args["batchsize"], shuffle=True)

    exp_name = args["filter"]
    if args["wandb"] == "true":
        wandb.init(project="poly", name=exp_name)
    
    directory_path = './saved_images/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    directory_path = './saved_images/{}/'.format(exp_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    weights_path = './saved_weights/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    weights_path = './saved_weights/{}/'.format(exp_name)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    prev_total_test_loss = 1000

    for epoch in range(args["num_epochs"]):
        model.train()
        total_loss = 0.0
        total_test_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if i % args["log_every"] == 0:
                print ('Epoch: [{} / {}], Iteration: [{} / {}], Train Loss: {}'.format(
                        epoch, args["num_epochs"], i, 
                        len(train_dataset) // args["batchsize"], 
                        loss.item()
                    ))
                save_image_grid(inputs, directory_path, 'inputs_{}_{}'.format(epoch, i))
                save_image_grid(targets, directory_path, 'targets_{}_{}'.format(epoch, i))
                save_image_grid(outputs, directory_path, 'outputs_{}_{}'.format(epoch, i))

                if args["wandb"] == "true":
                    wandb.log({"Train Loss": loss.item()})
            
        print ('End of the Epoch ...')
        print ('Epoch: [{} / {}], Train Loss: {}'.format(
                epoch, args["num_epochs"],  
                total_loss / len(train_loader)
            ))
        if args["wandb"] == "true":
            wandb.log({"Mean Train Loss": total_loss / len(train_loader)})
        
        model.eval()
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_test_loss += loss.item()

        if (abs(prev_total_test_loss - total_test_loss) * 100 / total_test_loss) < 0.1:
            break

        print ('Epoch: [{} / {}], Val Loss: {}'.format(
                epoch, args["num_epochs"], 
                total_test_loss / len(test_loader)
            ))
        if args["wandb"] == "true":
            wandb.log({"Mean Val Loss": total_test_loss / len(test_loader)})
        
        save_image_grid(inputs, directory_path, 'inputs_val_{}'.format(epoch))
        save_image_grid(targets, directory_path, 'targets_val_{}'.format(epoch))
        save_image_grid(outputs, directory_path, 'outputs_val_{}'.format(epoch))

        if prev_total_test_loss > total_test_loss:
            torch.save(model.state_dict(), './{}/model.pth'.format(weights_path))
        prev_total_test_loss = total_test_loss

if __name__ == '__main__':
    # Train the model
    train_model()
