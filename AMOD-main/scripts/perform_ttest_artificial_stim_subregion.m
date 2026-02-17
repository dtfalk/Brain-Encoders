% This script runs a regression on y = predicted activations for artificial stimuli in encoding models of amygdala subregions (AStr, CM, LB, SF) on x = target_roi, target_subject, RGB, high and low frequencies

addpath(genpath('Github'))

%%

t = readtable('artificial_stim_data_subregions_z.csv');

t.target_roi=categorical(t.target_roi);

t.target_subject=categorical(t.target_subject);

%% models for each subject

clear betas;

regions = {'''AStr''','''CM''','''LB''','''SF'''};

for r=1:4;

    for s = 1:20

        tsub = t(t.enc_subject==s,:);
        
        tsub.on_vs_off=double(categorical(regions(r))==tsub.target_roi);
        
        tsub.on_vs_off(tsub.on_vs_off==0)=-1;

        lme_sub = fitlme(tsub, [strrep(regions{r},'''','') '_enc_pred ~ on_vs_off  + target_subject + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z']);

        betas(s,r)=lme_sub.Coefficients.Estimate(end);

    end

end

% group inference ttest
[h, p, ci, st] = ttest(betas)

barplot_columns(betas(:,:))