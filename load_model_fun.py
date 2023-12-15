import torch
import modelMp
import cv2
import random
from torchvision import datasets, transforms

import torch.nn as nn

def make_noise_sample_p(image, t=10, include_last = True, include_first = True):
    li = []
    R = torch.randint(2, size=image.shape, dtype=torch.bool)
    if include_last:
        noisy_image = torch.logical_xor(image, R)
        li.append(noisy_image.clone()) 
    ctm = None
    for _ in range(t):
        c = torch.nonzero(R, as_tuple=True)
        ct = c[0].shape[0]
        if not ctm:
            ctm = ct
        if not include_first and ctm//t + 1 > ct:
            break
        noise_index = torch.randperm(ct)[:min(ct, ctm//t + 1)] 
        R[0, c[1][noise_index], c[2][noise_index]] = 0
        
        noisy_image = torch.logical_xor(image, R)
        li.append(noisy_image.clone()) 
    li.reverse()
    return li

def make_noise_sample(image,tot,t,M,R):
    B = R.clone()
    B[M>(t/tot)] = 0
    return torch.logical_xor(image, B)

def make_noise_sample_s(image, t=10, include_last = True, include_first = True):
    li = []
    M = torch.rand(size=image.shape)
    #M = M**2
    R = torch.randint(2, size=image.shape, dtype=torch.bool)
    if include_first:
        li.append(image)
    for n in range(1,t):
        noisy_image = make_noise_sample(image, t, n, M, R)
        li.append(noisy_image.clone()) 
    if include_last:
        li.append(torch.logical_xor(image, R))
    return li

def make_noise_sample_v(image, tot, t, M,R):
    B = R.clone()
    B[M>(2.7**(t/tot)/(tot))] = 0
    return torch.logical_xor(image, B)

def make_noise_sample_s_v(image, t=10, include_last = True, include_first = True):
    li = []
    if include_first:
        li.append(image)
    for n in range(1,t+1):
        if include_last or n<=t:
            M = torch.rand(size=image.shape)
            R = torch.randint(2, size=image.shape, dtype=torch.bool)
            noisy_image = make_noise_sample_v(image, t, n, M, R)
            li.append(noisy_image.clone()) 
            image = noisy_image
    return li

def model_visual_run(modelname, test_t = 5, itm = 1, itm_rescale = True, img_seed = None, t_seed =None, dif_t = 10):
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
        for _ in range(32):
            rrr = random.randint(0, len(selected)-1)
            print(rrr, end=', ')
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

    output[output<0.005] = 0

    output_np = [
        (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        for yy in output]
    output_np2 = [
        (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        for yy in input_tensor]
    output_np3 = [
        (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        for yy in og_img]
    output_np.extend(output_np2)
    output_np.extend(output_np3)

    num_images = len(output_np)
    num_cols = 16
    num_rows = (num_images // num_cols)
    border_color = (255, 255, 255) 
    border_size = 1 
    canvas = None
    for i in range(num_rows):
        row_images = output_np[i * num_cols: (i + 1) * num_cols] 
        bordered_images = []
        for img in row_images:
            bordered_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                            cv2.BORDER_CONSTANT, value=border_color)
            bordered_images.append(bordered_img)

        row_combined = cv2.hconcat(bordered_images)

        if canvas is None:
            canvas = row_combined
        else:
            canvas = cv2.vconcat([canvas, row_combined])

    cv2.namedWindow('Combined Images with Borders', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Images with Borders', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()