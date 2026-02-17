% Amygdala subregions (CM = centromedial, SF = superficial, AStr = amygdala to striatal, LB = laterobasal)

addpath(genpath('Github'))
load('500_days_of_summer_fc7_features.mat')
lendelta = size(video_imageFeatures, 1);

for i = 1:size(video_imageFeatures, 2)

    tmp = conv(double(video_imageFeatures(:,i)), spm_hrf(1));

    X(:,i) = tmp(1:lendelta);

end

subjects = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};

rois = {'Amygdala_CM_'  'Amygdala_SF_'  'Amygdala_AStr_'  'Amygdala_LB_'};

for s = 1:length(subjects)

    dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-' subjects{s} '/func/sub-' subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz']);

    for r = 1:length(rois)

        masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),rois{r}));
        disp('masked_dat done')

        features = resample(double(X),size(masked_dat.dat,2),size(X,1));
        disp('features done')

        [~,~,~,~,b] = plsregress(features,masked_dat.dat',20);

        save(['beta_sub-' subjects{s} '_' rois{r} 'fc7_invert_imageFeatures.mat'], 'b', '-v7.3')

    end

end