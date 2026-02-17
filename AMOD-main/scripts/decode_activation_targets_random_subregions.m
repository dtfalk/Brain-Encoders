% This script tests the discriminability of the predicted activations for the artificial stimuli
% generated from randomly selected voxels from the amygdala encoding models

addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')
load netTransfer_20cat.mat

%%

subject = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};
roi = {'region_1' 'region_2' 'region_3' 'region_4'};

%%

for it = 1:1000
    
    % feed artificial stim into ANN for predicted activations
    
    c=1;

    % encoding models
    
    for s = 1:length(subject)

        for g = 1:length(roi)
            
            table = readtable(strrep(['/home/data/eccolab/AMax/encoding_models/betas/random_subregion_betas/meanbeta_' roi{g} '_sub-' subject{s} '_amyFearful_fc7_invert_imageFeatures_iteration_' num2str(it) '.csv'],'_region_','_region'));

            betas(s,:,g) = table.Var1;

        end

    end

    subj_inds=1:20;

    for s = 1:length(subject)

        for g = 1:length(roi)

            % artificial stimuli

            imgName = ['/home/data/eccolab/Code/ActMax-Optimizer-Dev/tmp/Best_Img_emonet_genfc6_actfc7_S' sprintf('%02d',str2double(subject{s})) '_Amygdala_random_' roi{g} '_iteration_' num2str(it) '.png'];

            image_name{c} = imgName;

            img = imread([imgName]);
            
            img = img(100:end-100,100:end-100,:); %crop
            
            % get predicted activation for artificial stim in ANN
            acts(c,:) = activations(netTransfer, readAndPreprocessImage(img),'fc7');

            % get predicted activations for artificial stim in encoding models
            enc_prediction(c,:) = squeeze(acts(c,:))*squeeze(betas(s,:,:));
            
            enc_prediction_avg(s,c,:) = squeeze(acts(c,:))*squeeze(mean(betas(subj_inds~=s,:,:)));

            subj_full{c}=subject{s};
            
            roi_full{c}=roi{g};

            c=c+1;

        end
    
    end

    %% supervised learning, predict activation target

    Y = zscore(condf2indic(categorical(roi_full)));

    kinds = double(categorical(subj_full));
    
    for k=1:max(kinds)

        train = kinds~=k;
        
        test = ~train;

        [~,~,~,~,b] = plsregress(double(enc_prediction(train,:)),Y(train,:),4);
        
        yhat(kinds==k,:)=[ones(length(find(kinds==k)),1) double(enc_prediction(kinds==k,:))]*b;

    end

    [~, pred_cat]=max(yhat,[],2);
    
    [~, true_cat]=max(Y,[],2);
    
    accuracy(it) = mean(pred_cat==true_cat);
    
    % [k,stats] = cluster_confusion_matrix(pred_cat,true_cat,'labels',roi','method', 'ward','dofig','pairwise');
    
    % res = binotest(stats.optimalY(pred_cat)==stats.optimalY(true_cat),mean(crosstab(true_cat)./length(true_cat)));

end

