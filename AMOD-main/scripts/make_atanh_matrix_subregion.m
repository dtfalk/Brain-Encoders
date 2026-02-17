% This script takes the amygdala atanh matrix and masks by subregions to get predicted activations for each subregion

addpath(genpath('/home/data/eccolab/Code/GitHub'))

rois = {'Amygdala_LB_' 'Amygdala_SF_' 'Amygdala_CM_' 'Amygdala_AStr_'};

load('/home/gjang/amygdala_fc7_invert_imageFeatures_output_matrix_atanh.mat')

atanh_matrix(atanh_matrix==0) = NaN

dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-1/func/sub-1_task-500daysofsummer_bold_blur_censor.nii.gz']);

masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),{'Amy'}));

masked_dat = replace_empty(masked_dat);

amy_bin = double(masked_dat.dat(:,1)~=0);

for r = 1:length(rois)

    masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),rois{r}));

    masked_dat = replace_empty(masked_dat);

    rois_bin(r,:) = masked_dat.dat(amy_bin==1,1)~=0;

    rois_inds{r} = find(rois_bin(r,:));

    atanh_subregion = atanh_matrix(:,rois_bin(r,:)==1);

    avg_atanh_subregion(:,r) = mean(atanh_subregion,2);

end

save(['amygdala_subregions_atanh.mat'],'avg_atanh_subregion')



