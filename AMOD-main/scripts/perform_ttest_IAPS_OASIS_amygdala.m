% This script runs the regression of all the predictor variables on amygdala (and visual cortex) predictions separately in a one-sample ttest

addpath(genpath('Github'))

%% Average over IAPS and OASIS (for more compact set of results)

filenames = {'OASIS_data_amygdala_z' 'IAPS_data_amygdala_z'};

for f=1:length(filenames)
    
    t = readtable([filenames{f} '.csv']);

    for s = 1:20
        
        tsub = t(t.subject==s,:);
        
        lme_sub = fitlme(tsub,'amy_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z ');
        betas(f,s,:)=lme_sub.Coefficients.Estimate;

        lme_sub_vc = fitlme(tsub,'vc_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z ');
        betas_vc(f,s,:)=lme_sub_vc.Coefficients.Estimate;
    
    end

end

%% One-sample t-tests

avg_betas = mean(betas);
%avg_betas_vc = mean(betas_vc);

[h, p, ci, st] = ttest(squeeze(avg_betas))
%[h, p, ci, st]=ttest(squeeze(avg_betas_vc))

barplot_columns(squeeze(avg_betas),'dolines')
%barplot_columns(squeeze(avg_betas_vc),'dolines')

