% This script is for identifying associations between voxelwise activity in the amygdala encoding 
% models and valence, arousal and their interaction, using the IAPS database

addpath(genpath('GitHub'))
load netTransfer_20cat.mat
load('500_days_of_summer_fc7_features.mat')
lendelta = size(video_imageFeatures, 1);

%%
subject = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};

%%
% get IAPS images
% read number of IAPS images based on runs into list
imgList= dir(['/home/data/eccolab/IAPS/*.jpg']);

num_run = 1:length(imgList);

for f = num_run

    % get image name for artificial stimuli
    imgName = ([imgList(f).folder filesep imgList(f).name]);
    
    % read artificial stim as images
    img = imread([imgName]);
    
    % valence 
    load(strrep(imgName,'.jpg','_valence.mat'));
    val(f) = X;
    
    % arousal 
    load(strrep(imgName,'.jpg','_arousal.mat'));
    arousal(f) = X;
    
    % get predicted activation for artificial stim in ANN
    acts(f,:) = activations(netTransfer, readAndPreprocessImage(img),'fc7');

end

%%
for s = 1:length(subject)

    dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-' subject{s} '/func/sub-' subject{s} '_task-500daysofsummer_bold_blur_censor.nii.gz']);

    masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),{'Amy'})); 

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

    % get the predicted activations from encoding models
    enc_prediction =[ones(height(acts),1) squeeze(acts)]*squeeze(b);

    % correlate predicted activations with valence, arousal and their interaction 
    [voxelwise_correlations,p] = corr(squeeze(enc_prediction),[zscore(val'), zscore(arousal'), zscore(val').*zscore(arousal')]);
    Rmat(s,~excluded_voxels(s,:),:) = voxelwise_correlations;

end


%%
regression_object=masked_dat;
regression_object.removed_voxels = ~any(excluded_voxels==0)';
regression_object.removed_images = 0;
regression_object.dat = squeeze(atanh(Rmat(:,any(excluded_voxels==0),1)))';
table(threshold(ttest(regression_object),.005,'UNC'));
regression_object.fullpath = 'voxelwise_correlations_IAPS_valence_amygdala.nii';
regression_object.write;
%%
regression_object=masked_dat;
regression_object.removed_voxels = ~any(excluded_voxels==0)';
regression_object.removed_images = 0;
regression_object.dat = squeeze(atanh(Rmat(:,any(excluded_voxels==0),2)))';
table(threshold(ttest(regression_object),.005,'UNC'));
regression_object.fullpath = 'voxelwise_correlations_IAPS_arousal_amygdala.nii';
regression_object.write;
%%
regression_object=masked_dat;
regression_object.removed_voxels = ~any(excluded_voxels==0)';
regression_object.removed_images = 0;
regression_object.dat = squeeze(atanh(Rmat(:,any(excluded_voxels==0),3)))';
table(threshold(ttest(regression_object),.005,'UNC'));
regression_object.fullpath = 'voxelwise_correlations_IAPS_valence_arousal_interaction_amygdala.nii';
regression_object.write;