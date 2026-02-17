#!/bin/bash
#SBATCH --account=default
#SBATCH --partition=week-long
#SBATCH --nodelist=gpu1
#SBATCH --time=6-00:00:00
#SBATCH --mem=64GB  #ram
#SBATCH --gpus-per-node=2

eval "$(conda shell.bash hook)"
conda activate AMOD
cd /home/data/eccolab/Code/ActMax-Optimizer-Dev/

#iterate over subjects to generate artificial stimuli to optimize brain activity in ROIs: Amygdala, IT, VC (averaged across voxels = mean betas)
##python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/pkragel/beta_csv/fc7/amygdala/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_amygdala_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amygdala_randinit
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/amygdala/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_amygdala_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amygdala_randinit_run19 --output_folder output/amygdala/actfc7
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/itcortex/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_itcortex_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_itcortex_randinit_run19 --output_folder output/itcortex/actfc7
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/visualcortex/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_visualcortex_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_visualcortex_randinit_run19 --output_folder output/visualcortex/actfc7

#iterate over subjects to generate artificial stimuli to optimize brain activity in Amygdala subregions ROIs: AStr, CM, LB, SF (averaged across voxels = mean betas)
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/amygdala_subregions_GJ/AStr/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_Amygdala_AStr_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amyAStr_randinit_run19 --output_folder output/amygdala_subregions_GJ/AStr/actfc7
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/amygdala_subregions_GJ/CM/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_Amygdala_CM_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amyCM_randinit_run20 --output_folder output/amygdala_subregions_GJ/CM/actfc7
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/amygdala_subregions_GJ/LB/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_Amygdala_LB_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amyLB_randinit_run7 --output_folder output/amygdala_subregions_GJ/LB/actfc7
#python3 ActMax.py --act_layer .Conv2dConv_6 --encoding_filename /home/data/eccolab/Code/ActMax-Optimizer-Dev/encoding_models/betas/fc7/amygdala_subregions_GJ/SF/mean_b/meanbeta_sub-${SLURM_ARRAY_TASK_ID}_Amygdala_SF_fc7_invert_imageFeatures.csv --output_filename emonet_genfc6_actfc7_sub${SLURM_ARRAY_TASK_ID}_meanbetas_amySF_randinit_run22 --output_folder output/amygdala_subregions_GJ/SF/actfc7
