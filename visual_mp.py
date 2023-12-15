from load_model_fun import model_visual_run
#test_t t seleceted for input noise
#itm: model iteration
#itm_rescale: scale output value > 0.5 to 1 as input
#dif_t: total time step
#img_seed: seed for selecting sample test image
#t_seed: seed for all torch processing
#modelname, test_t = 5, itm = 1, itm_rescale = True, img_seed = None, t_seed =None, dif_t = 10
model_visual_run("bmmodelMp1B1n2.pth", test_t=10, itm=10, itm_rescale=True)
