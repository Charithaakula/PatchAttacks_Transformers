#create folders of the models in alloutputs if not present already

#take array of args and generate lines of the form 
#sbatch eval_attacks.sh deit3 /scratch/aaj458/data/ImageNet/val 100 20 300 0.01 8 
#sleep 10
#which will be written to a bash script which we will run to generate mutliple batch scripts at a time
import os

file = open("myeval.sh", "w")

models = ['swin224', 'beit224']
datapaths = ['/imagenet/']
num_iters = [32]
max_patches = [10, 20]
num_images = 1000
lrs = [0.1, 0.01]
patch_sizes = [8, 16]
start_idxs = [0]

for model in models:
    #create folder in alloutputs if it doesn't exists for this modelname
    if not os.path.exists('alloutputs/' + model):
    # Create the directory
        os.makedirs('alloutputs/' + model)
    for datapath in datapaths:
        for num_iter in num_iters:
            for max_patch in max_patches:                
                    for lr in lrs:
                        for patch_size in patch_sizes:
                            for start_idx in start_idxs:                                
                                s = 'sbatch evalattacks3.sh {} {} {} {} {} {} {} {}\n'                                                                
                                s = s.format(model,datapath,num_iter,max_patch,num_images,lr,patch_size,start_idx)
                                #print(s)
                                file.write(s)                              
                                file.write('sleep 2\n')

file.close()
