% These script gets the predicted activations for IAPS and OASIS images from encoding models, and also valence and arousal ratings

addpath(genpath('/home/data/eccolab/Code/GitHub'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
load netTransfer_20cat.mat

%%

subject = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'};
roi = {'amy'  'AStr' 'CM' 'LB' 'SF' 'vc'};

%% encoding models

c=1;

for s = 1:length(subject)

    for g = 1:length(roi)

        table = readtable(['/home/data/eccolab/AMax/encoding_models/betas/compare_images_mean_b/meanbeta_sub-' subject{s} '_' roi{g} '_fc7_invert_imageFeatures.csv']); % use mean_betas

        betas(s,:,g) = table.Var1;

    end

end

%% get IAPS and OASIS predicted activations, valence, and arousal 

subj_inds=1:20;

% imgList= dir(['/home/data/eccolab/IAPS/*.jpg']);
imgList= dir(['/home/data/eccolab/OASIS/Images/*.jpg']);

num_run = 1:length(imgList);

for f = num_run

    imgName = ([imgList(f).folder filesep imgList(f).name]);
    img = imread([imgName]);
    
    load(strrep(imgName,'.jpg','_valence.mat'));
    val(f)=X;
    
    load(strrep(imgName,'.jpg','_arousal.mat'));
    arousal(f)=X;

    acts(f,:) = activations(netTransfer, readAndPreprocessImage(img),'fc7');

    for s = 1:length(subject)

        enc_prediction(c,:) = squeeze(acts(f,:))*squeeze(betas(s,:,:));
        enc_prediction_avg(c,:) = squeeze(acts(f,:))*squeeze(mean(betas(subj_inds~=s,:,:)));

        subj_full{c}=subject{s};
        val_full(c)=val(f);
        arous_full(c)=arousal(f);

        c=c+1;

    end

end

%% correlate predicted activations with arousal

ap = prctile(arousal,[0:10:100]);

for s=1:length(subject)
    
    for a=2:length(ap)
        
        data_to_plot(s,a-1,:) = mean(enc_prediction(arous_full > ap(a-1) & arous_full < ap(a) & strcmp(subj_full,subject{s}),:));
        data_to_plot_subj_independent(s,a-1,:) = mean(enc_prediction_avg(arous_full > ap(a-1) & arous_full < ap(a) & strcmp(subj_full,subject{s}),:));
    
    end

    arousal_corr(s,:)=corr(enc_prediction(strcmp(subj_full,subject{s}),:),arous_full(strcmp(subj_full,subject{s}))');

end

[p tbl st]=friedman(squeeze(data_to_plot(:,:,1)))

multcompare(st)

barplot_columns(data_to_plot(:,:,1))

%% correlate predicted activations with valence

vp = prctile(val,[0:10:100]);

for s=1:length(subject)
    
    for v=2:length(vp)

        data_to_plot(s,v-1,:) = mean(enc_prediction(val_full > vp(v-1) & val_full < vp(v) & strcmp(subj_full,subject{s}),:));
        data_to_plot_subj_independent(s,v-1,:) = mean(enc_prediction_avg(val_full > vp(v-1) & val_full < vp(v) & strcmp(subj_full,subject{s}),:));
    
    end

    valence_corr(s,:)=corr(enc_prediction(strcmp(subj_full,subject{s}),:),val_full(strcmp(subj_full,subject{s}))');

end

[p tbl st]=friedman(squeeze(data_to_plot(:,:,1)))

multcompare(st)

barplot_columns(data_to_plot(:,:,1))

[h p ci st]=ttest(squeeze(data_to_plot(:,:,1))*[-1 -4/5 -3/5 -2/5 -1/5 1/5 2/5 3/5 4/5 1]') 

%%

figure;

for r=1:2
    
    subplot(2,1,r);bar(squeeze(nanmean(data_to_plot(:,:,r))));

end