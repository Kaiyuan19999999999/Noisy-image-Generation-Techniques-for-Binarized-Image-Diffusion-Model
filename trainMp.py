import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
from modelMp import binmnistmodel, make_noise_sample_s
#from load_model_fun import make_noise_sample_s_v as make_noise_sample_s

if __name__ == "__main__":
    model = binmnistmodel()
    model_file = "bmmodelMp2B1n.pth"
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print(model_file, " Model loaded successfully!")
    else:
        print(f"File '{model_file}' does not exist. Automatically creating when finsh")

    transform = transforms.Compose([
        #transforms.Resize((28, 28)),  # Resize images to 28x28
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.bernoulli(x))
    ])

    ogd = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset = []
    for _ in range(5000):
        rrr = random.randint(0, len(ogd)-1)
        image = ogd[rrr][0]
        li = make_noise_sample_s(image)
        li2 = make_noise_sample_s(image)
        #train_dataset.append((image, image))
        for n in range(len(li)-1):
            train_dataset.append((li[n+1], li[n]))
            train_dataset.append((li2[n+1], li2[n]))


    print(len(train_dataset))
    #train_dataset = ogd

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    num_epochs = 3
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, doutput = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            dif = torch.zeros(inputs.shape)
            dif[torch.logical_xor(doutput, inputs)] = 1
            c1 = dif
            # c1 = torch.square(dif).view(inputs.shape[0], inputs.shape[1], -1)
            # ma = torch.max(c1, dim=2, keepdim=True)[0]
            # #c1[torch.nonzero(ma==0)] = 1
            # ma[ma == 0] = 1
            # c1 = c1/ma
            # c1 = c1.view(inputs.shape)
            loss = criterion(torch.mul(outputs, c1), torch.mul(doutput, c1))
        
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss:.4f}")
                running_loss = 0.0
        torch.save(model.state_dict(), model_file)

    print("Finished Training")
