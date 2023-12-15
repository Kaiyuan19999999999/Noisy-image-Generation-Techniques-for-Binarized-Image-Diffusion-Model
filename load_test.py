import torch
import modelMp
import random
from torchvision import datasets, transforms

import torch.nn as nn
from load_model_fun import make_noise_sample_p as make_noise_sample_s

def model_run(modelname, test_t = 5, itm = 1, itm_rescale = True, img_seed = None, t_seed =None, dif_t = 10):
    loaded_model = modelMp.binmnistmodel()
    loaded_model.load_state_dict(torch.load(modelname))

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.bernoulli(x))
        ])
    
    if t_seed: 
        torch.manual_seed(t_seed)
    if img_seed:
        random.seed(img_seed)

    if (test_t >= dif_t):
        input_tensor = torch.randn(32, 1, 28, 28)
        input_tensor = torch.where(input_tensor > 0, torch.tensor(1.0), torch.tensor(0.0))
        og_img = torch.zeros_like(input_tensor)
    else:
        selected = datasets.MNIST(root='./data', train=False, download=True, transform=transform)     
        og_img = []
        input_tensor = []
        
        #for _ in range(32):
        for _ in range(640):
            rrr = random.randint(0, len(selected)-1)
            #print(rrr, end=', ')
            og_img.append(selected[rrr][0])
            input_tensor.append( make_noise_sample_s(selected[rrr][0], t = dif_t, include_last=False) [test_t] ) #<<<<<<<<<
            
        input_tensor = torch.stack(input_tensor)
        og_img = torch.stack(og_img)


    print(input_tensor.shape) 

    criterion = nn.MSELoss()

    with torch.no_grad():
        output = input_tensor
        running_loss = 0
        for _ in range(itm): #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if itm_rescale:
                output[output>0.5] = 1 
            output = loaded_model(output)
            #output[output<0.01] = 0

        dif = torch.zeros(input_tensor.shape)
        dif[torch.logical_xor(og_img, input_tensor)] = 1
        c1 = torch.square(dif).view(input_tensor.shape[0], input_tensor.shape[1], -1)
        ma = torch.max(c1, dim=2, keepdim=True)[0]
        c1[torch.nonzero(ma==0)] = 1
        ma[ma == 0] = 1
        c1 = c1/ma
        c1 = c1.view(input_tensor.shape)
        loss = criterion(torch.mul(output, c1), torch.mul(og_img, c1))
        running_loss += loss.item()

        print(f"Loss: {running_loss:.4f}")
