# PatchAttacks_Transformers

## Intro:
Main changes are in 'patchattackgrad_multi.py'. Need to install necesary packages to run this locally

## Running attacks:
1. Run 'generate.py' by adding various options for all the test cases to be generated. 
2. 'generate.py' will create 'myeval.sh', running which will submit parallel sbatch scripts to run all the test cases. 

## Outputs:
1. The results will be stored in the alloutputs folder, with each model represented in a subfolder (e.g., alloutput/swin224 for all test cases belonging to the swin224 model).
2. Inside each model folder, a folder will be created for each experiment, with the name indicating all the inputs (e.g., mt_deit3_224_it_32_mp_10_ni_1000_lr_0.1_ps_8_0_eps_1.0). This folder will contain all the examples that were run, with a 0 or 1 indicating whether the attack was successful or not. A run.log file will also be stored in this folder, at the beginning of which the clean accuracy will be logged and at the end of which the robust accuracy will be logged.
