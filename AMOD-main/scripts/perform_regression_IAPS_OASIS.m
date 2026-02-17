% This script makes the surface plots for IAPS and OASIS images by running regression for each roi, calculates the piecewise amygdala activations by valence (neg, neutral, pos), and runs anovas for valence and interaction

%% regression analyses

%load subregions' predicted activations for IAPS and OASIS images from predict_activations_IAPS_OASIS.m

IAPS_activations = load('amygdala_subregion_encoding_output_IAPS.mat')
OASIS_activations = load('amygdala_subregion_encoding_output_OASIS.mat')

%load predictor variables

filenames = {'OASIS_data_amygdala_z' 'IAPS_data_amygdala_z'};

figure;cc=0;

for f=1:length(filenames)
    
    clear yhat
    
    t = readtable([filenames{f} '.csv']);

    if f==1

        t.AStr_enc_pred = double(OASIS_activations.enc_prediction(:,2));
        t.CM_enc_pred = double(OASIS_activations.enc_prediction(:,3));
        t.LB_enc_pred = double(OASIS_activations.enc_prediction(:,4));
        t.SF_enc_pred = double(OASIS_activations.enc_prediction(:,5));
    
    else
        
        t.AStr_enc_pred = double(IAPS_activations.enc_prediction(:,2));
        t.CM_enc_pred = double(IAPS_activations.enc_prediction(:,3));
        t.LB_enc_pred = double(IAPS_activations.enc_prediction(:,4));
        t.SF_enc_pred = double(IAPS_activations.enc_prediction(:,5));
    
    end

    rois = {'amy'  'AStr' 'CM' 'LB' 'SF' 'vc'};

    for r=1:6

        for s = 1:20

            tsub = t(t.subject==s,:);

            lme_sub = fitlme(tsub,[rois{r} '_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z']);

            betas(f,s,r,:)=lme_sub.Coefficients.Estimate;

            %calculate predicted yhat values using only val, arous, and val:arous from full regression equation for each subject for each IAPS image (20 yhats per image)
            yhat(s,r,:)= [ones(height(tsub),1) tsub.val_full_z tsub.arous_full_z tsub.val_full_z.*tsub.arous_full_z]*lme_sub.Coefficients.Estimate([1:3 end]);

        end

        yhat_plot = squeeze(mean(yhat(:,r,:)));

        xnodes = -2.5:.2:2.5;
        
        ynodes = -2:.2:2;

        g = gridfit(double(tsub.val_full_z),double(tsub.arous_full_z),yhat_plot,xnodes,ynodes, 'smoothness',1,'interp','triangle','regularizer','gradient');
        
        cc=cc+1;

        subplot(2,6,cc);

        hold on;

        surfc(-2.5:.2:2.5,-2:.2:2,g)
        
        view([-35 15])

        title([rois{r} ': ' filenames{f}(1:3)]);
        scatter3(double(tsub.val_full_z),double(tsub.arous_full_z),double(yhat_plot),100,'marker','.')
        axis tight
        colorbar; grid on; drawnow;pause(.2)

        % regression on neutral to positive
        
        t_positive = t(t.val_full_z>0,:);

        for s = 1:20

            tsub_pos = t_positive(t_positive.subject==s,:);

            lme_sub_pos = fitlme(tsub_pos,'amy_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z ');

            betas_pos(f,s,r,:)=lme_sub_pos.Coefficients.Estimate;

        end

        % regression on neutral to negative
        
        t_negative = t(t.val_full_z<0,:);

        for s = 1:20

            tsub_neg = t_negative(t_negative.subject==s,:);

            lme_sub_neg = fitlme(tsub_neg,'amy_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z ');

            betas_neg(f,s,r,:)=lme_sub_neg.Coefficients.Estimate;

        end

        % regression around neutral
        
        t_mid = t(abs(t.val_full_z)<1,:);

        for s = 1:20

            tsub_mid = t_mid(t_mid.subject==s,:);

            lme_sub_mid = fitlme(tsub_mid,'amy_enc_pred ~ 1  + val_full_z + arous_full_z + val_full_z:arous_full_z + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z ');

            betas_mid(f,s,r,:)=lme_sub_mid.Coefficients.Estimate;

        end

    end

end

%%

mean_betas = squeeze(mean(betas));

for r=1:length(rois)
    
    figure;cc=0;

    cc=cc+1;

    subplot(4,1,cc)
    
    barplot_columns(squeeze(mean_betas(:,r,2:end)),rois{r},'nofig');

    set(gca,'Xticklabel',lme_sub.CoefficientNames(2:end))
    
    title(rois{r})
    
    cc=cc+1;

    subplot(4,1,cc)
    
    barplot_columns(squeeze(squeeze(mean(betas_neg(:,:,r,2:end)))),['Neg ' rois{r}],'nofig');

    set(gca,'Xticklabel',lme_sub.CoefficientNames(2:end))

    cc=cc+1;

    subplot(4,1,cc)
    
    barplot_columns(squeeze(squeeze(mean(betas_mid(:,:,r,2:end)))),['Mid ' rois{r}],'nofig');

    set(gca,'Xticklabel',lme_sub.CoefficientNames(2:end))

    cc=cc+1;

    subplot(4,1,cc)
    
    barplot_columns(squeeze(squeeze(mean(betas_pos(:,:,r,2:end)))),['Pos ' rois{r}],'nofig');

    set(gca,'Xticklabel',lme_sub.CoefficientNames(2:end))

end


%% anova for val

betas_per_roi = squeeze(mean_betas(:,2:end-1,2));

t_subregions = table(categorical((1:20)'),betas_per_roi(:,1),betas_per_roi(:,2),betas_per_roi(:,3),betas_per_roi(:,4));

variablenames = ["subject","AStr","CM","LB","SF"];

meas =  table(categorical(variablenames(2:end))',VariableNames="ROI");

rm = fitrm(t_subregions,"Var2-Var5~1",WithinDesign=meas);

anova(rm)

multcompare(rm,'ROI')

%% anova for interaction (val:arousal)

betas_per_roi = squeeze(mean_betas(:,2:end-1,9));

t_subregions = table(categorical((1:20)'),betas_per_roi(:,1),betas_per_roi(:,2),betas_per_roi(:,3),betas_per_roi(:,4));

variablenames = ["subject","AStr","CM","LB","SF"];

meas =  table(categorical(variablenames(2:end))',VariableNames="ROI");

rm = fitrm(t_subregions,"Var2-Var5~1",WithinDesign=meas);

anova(rm)

multcompare(rm,'ROI')

%%

C = [125 11 11; 69 204 102; 252 135 60; 100 107 203;250 235 84;166 138 106]./255;

for r=1:6

    yhat_avg(r,:)= [ones(height(tsub),1) tsub.val_full_z tsub.arous_full_z tsub.val_full_z.*tsub.arous_full_z]*squeeze(mean(mean_betas(:,r,[1:3 end])));

    for i=1:3; xC(:,i)=log10(logspace(C(r,i),1,32));end
    
    xnodes = -2.5:.2:2.5;
    
    ynodes = -2:.2:2;

    g = gridfit(double(tsub.val_full_z),double(tsub.arous_full_z),yhat_avg(r,:)',xnodes,ynodes, 'smoothness',1,'interp','triangle','regularizer','gradient');

    subplot(2,3,r);

    hold on;

    h=surfc(-2.5:.2:2.5,-2:.2:2,g);
    
    h(1).EdgeAlpha=.1;
    
    h(2).LineWidth = 2;
    
    clim([0 .15])
    
    view([-35 15])
    
    colormap(gca,flipud(xC))

    title([rois{r}]);

    axis tight
    
    colorbar; grid on; drawnow;pause(.2)

end


