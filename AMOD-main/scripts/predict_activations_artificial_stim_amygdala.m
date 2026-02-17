% These script gets the predicted activations for generated artificial stimuli from the encoding models (amy, vc)

addpath(genpath('/home/data/eccolab/Code/GitHub'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
load netTransfer_20cat.mat

%%

subject = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};
roi = {'amy' 'vc'};

%% encoding models

c=1;

for s = 1:length(subject)

    for g = 1:length(roi)

        table = readtable(['/home/data/eccolab/AMax/encoding_models/betas/compare_images_mean_b/meanbeta_sub-' subject{s} '_' roi{g} '_fc7_invert_imageFeatures.csv']);

        betas(s,:,g) = table.Var1;

    end

end

%% get predicted activations for artificial stim

subj_inds=1:20;

imgList= dir(['/home/data/eccolab/AMax/online_behavior/Pavlovia/artificial_stim_used_v2-4_excluding_patterns/*.png']);

num_run = 1:length(imgList);

% make empty arrays to contain target subject and target roi columns
target_subject_full = cell(length(num_run),1);
target_roi_full = cell(length(num_run),1);

for f = num_run

    imgName = ([imgList(f).folder filesep imgList(f).name]);

    splitName = split(imgName,"_");
    
    target_subject = splitName(9,:);
    
    target_roi = splitName(10,:);

    img = imread([imgName]);
    
    % activation for artificial stim in ANN
    acts(f,:) = activations(netTransfer, readAndPreprocessImage(img),'fc7');

    target_subject_full{f} = target_subject;
    
    target_roi_full{f} = target_roi;

    % predicted activations for artificial stim from encoding models
    
    for s = 1:length(subject)

        enc_prediction(c,:) = squeeze(acts(f,:))*squeeze(betas(s,:,:));
        
        enc_prediction_avg(c,:) = squeeze(acts(f,:))*squeeze(mean(betas(subj_inds~=s,:,:)));

        subj_full{c}=subject{s};
        
        c=c+1;

    end

end

save(['enc_prediction_amy_vc_artStim.mat'],'enc_prediction');
