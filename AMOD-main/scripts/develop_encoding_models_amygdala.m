% This script performs partial least squares regression on Y = amygdala voxels ~ X = features to get betas for encoding models, and performs 5-fold cross-validation to evaluate model performance

addpath(genpath('Github'))
load('500_days_of_summer_fc7_features.mat')
lendelta = size(video_imageFeatures, 1);

subjects = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};

for s = 1:length(subjects)

    dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-' subjects{s} '/func/sub-' subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz']);

    masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),{'Amy'}));
    % masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('glasser'),{'Ctx_V1', 'Ctx_V2', 'Ctx_V3'})); % early visual cortex (V1-V3)
    % masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('glasser'),{'Ctx_TE2', 'Ctx_TF'})); % inferotemporal cortex (TE2a, TE2p, TF)
    disp('masked_dat done')

    excluded_voxels(s,:) = masked_dat.removed_voxels

    features = resample(double(video_imageFeatures),size(masked_dat.dat,2),lendelta);
    disp('resample features done')

    % Convolute features to match time delay of hemodynamic BOLD data
    for i = 1:size(features, 2)

        tmp = conv(double(features(:,i)), spm_hrf(1));

        conv_features(:,i) = tmp(:);

    end

    % Match length of features to length of BOLD data
    timematched_features = conv_features(1:size(masked_dat.dat,2),:);
    disp('timematched_features done')

    % Extract regression coefficients (betas) for encoding models
    [~,~,~,~,b] = plsregress(timematched_features,masked_dat.dat',20);
    disp('beta done')

    kinds = crossvalind('k',length(masked_dat.dat),5);
    disp('kinds done')

    clear yhat pred_obs_corr diag_corr conv_features

    % 5-fold cross-validation
    for k=1:5

        [xl,yl,xs,ys,beta_cv,pctvar] = plsregress(timematched_features(kinds~=k,:), masked_dat.dat(:,kinds~=k)', min(20,size(masked_dat.dat,1)));
        disp('plsregress done')

        yhat(kinds==k,:)=[ones(length(find(kinds==k)),1) timematched_features(kinds==k,:)]*beta_cv;
        disp('yhat done')

        pred_obs_corr(:,:,k)=corr(yhat(kinds==k,:), masked_dat.dat(:,kinds==k)');
        disp('pred_obs_corr done')

        diag_corr(k,:)=diag(pred_obs_corr(:,:,k));
        disp('diag_corr done')

    end

    mean_diag_corr = mean(diag_corr)

    save(['sub-' subjects{s} '_amygdala_fc7_invert_imageFeatures_output.mat'], 'mean_diag_corr', '-v7.3')

    save(['beta_sub-' subjects{s} '_amygdala_fc7_invert_imageFeatures.mat'], 'b', '-v7.3')

end
