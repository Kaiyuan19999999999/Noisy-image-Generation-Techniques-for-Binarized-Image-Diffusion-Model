from load_model_fun import make_noise_sample_s, make_noise_sample_s_v, make_noise_sample_p
import cv2
import torch
import random
from torchvision import datasets, transforms

def model_visual_run(test_t = 5, itm = 1, itm_rescale = True, img_seed = None, t_seed =None, dif_t = 10):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.bernoulli(x))
        ])
    
    if t_seed: 
        torch.manual_seed(t_seed)
    if img_seed:
        random.seed(img_seed)
        

    selected = datasets.MNIST(root='./data', train=False, download=True, transform=transform)     
    og_img = []
    input_tensor = []
    input_tensor2 = []
    input_tensor3 = []
    #for _ in range(32):
    for _ in range(4):
        rrr = random.randint(0, len(selected)-1)
        print(rrr, end=', ')
        og_img.append(selected[rrr][0])
        input_tensor.extend( make_noise_sample_s(selected[rrr][0], t = dif_t, include_last=True))
        input_tensor2.extend( make_noise_sample_s_v(selected[rrr][0], t = dif_t, include_last=True) )
        input_tensor3.extend( make_noise_sample_p(selected[rrr][0], t = dif_t, include_last=True) )
        
    input_tensor = torch.stack(input_tensor)
    input_tensor2 = torch.stack(input_tensor2)
    input_tensor3 = torch.stack(input_tensor3)
    #og_img = torch.stack(og_img)


    print(input_tensor.shape) 


    output_np = [
        (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        for yy in input_tensor]
    output_np2 = [
        (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        for yy in input_tensor2]
    output_np3 = [
         (yy.detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
         for yy in input_tensor3]
    output_np.extend(output_np2)
    output_np.extend(output_np3)

    num_images = len(output_np)
    num_cols = 11
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

if __name__ == "__main__":
    model_visual_run()
