% This script generates the t-SNE plot, clustering solution, and confusion matrix for generated artificial stimuli based on their predicted activations in encoding models

addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')
load netTransfer_20cat.mat

%%

subject = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};
roi = {'amy' 'it' 'vc' 'AStr' 'CM' 'LB' 'SF'};

%% feed artificial stim into ANN for predicted activations

c=1;

% encoding models

for s = 1:length(subject)

    for g = 1:length(roi)

        table = readtable(['/home/data/eccolab/AMax/encoding_models/betas/compare_images_mean_b/meanbeta_sub-' subject{s} '_' roi{g} '_fc7_invert_imageFeatures.csv']); % use mean_betas

        betas(s,:,g) = table.Var1;

    end

end

subj_inds=1:20;

for s = 1:length(subject)

    for g = 1:length(roi)

        % artificial stimuli
        
        imgList= dir(['/home/data/eccolab/AMax/online_behavior/Pavlovia/artificial_stim_used_v2-4/emonet_fc7_sub' subject{s} '_' roi{g} '_run*.png']);
        
        num_run = 1:length(imgList);

        for f = num_run
            
            imgName = ([imgList(f).folder filesep imgList(f).name]);
            
            image_name{c} = imgName;

            img = imread([imgName]);
            
            I{c}=imresize(flipud(img),[512 512]);
            
            % activations by artificial stim in ANN
            acts(c,:) = activations(netTransfer, readAndPreprocessImage(img),'fc7');

            % predicted activations for artificial stim in encoding models
            enc_prediction(c,:) = squeeze(acts(c,:))*squeeze(betas(s,:,:));
            
            enc_prediction_avg(s,c,:) = squeeze(acts(c,:))*squeeze(mean(betas(subj_inds~=s,:,:)));

            image_mat(c,:) = img(:);
            
            subj_full{c}=subject{s};
            
            roi_full{c}=roi{g};
            
            run_full(c)=f;

            c=c+1;

        end

    end

end

%% create RMDS

RDM_pixel = 1-corr(double(image_mat'));

RDM_fc7 = 1-corr(double(acts'));

RDM_subj = 1- corr(condf2indic(categorical(subj_full))');

RDM_roi = 1- corr(condf2indic(categorical(roi_full))');

RDM_enc = 1- corr(double(enc_prediction'));

%% supervised learning, predict activation target; clustering solution and confusion matrix

Y = zscore(condf2indic(categorical(roi_full)));

kinds = double(categorical(subj_full));

for k=1:max(kinds)

    train = kinds~=k;
    
    test = ~train;

    [~,~,~,~,b] = plsregress(double(enc_prediction(train,:)),Y(train,:),7);

    yhat(kinds==k,:)=[ones(length(find(kinds==k)),1) double(enc_prediction(kinds==k,:))]*b;

end

[~, pred_cat]=max(yhat,[],2);

[~, true_cat]=max(Y,[],2);

[k,stats] = cluster_confusion_matrix(pred_cat,true_cat,'labels',roi','method', 'ward','dofig','pairwise');

res = binotest(stats.optimalY(pred_cat)==stats.optimalY(true_cat),mean(crosstab(true_cat)./length(true_cat)));

%% tsne plot

Y = tsne(enc_prediction);

Y_avg = tsne(squeeze(mean(enc_prediction_avg)));

figure;

plotOfImages = gscatter(Y(:,1),Y(:,2),roi_full',hsv(7));

axis tight;clc


%% compute chance accuracy for clustering

[phat, ndist] = permutation_test_multiway_clusters(pred_cat, true_cat);

mean(ndist)

std(ndist)