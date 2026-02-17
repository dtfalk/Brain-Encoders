% This script is for visualizing the associations between voxelwise activity in the amygdala encoding 
% models and valence, arousal and their interaction (Figure 5)

addpath(genpath('GitHub'))

%% arousal
% load niftis with correlation maps from OASIS for arousal
tdat=fmri_data('voxelwise_correlations_OASIS_arousal_amygdala.nii');

% Fisher transform correlation maps 
tdat.dat = atanh(tdat.dat);

% load niftis with correlation maps from IAPS for arousal
dat=fmri_data('voxelwise_correlations_IAPS_arousal_amygdala.nii');

% average Fisher transformed IAPS and OASIS correlation maps 
dat.dat = (atanh(dat.dat)+tdat.dat)/2;

% do a ttest on the averaged correlation maps
t = ttest(dat);

orthviews(t);

% threshold and save
table(region(threshold(t,.05,'UNC')));
t.fullpath = 'voxelwise_correlations_arousal_amygdala.nii';
t.write;

%% valence
% load niftis with correlation maps from OASIS for valence
tdat=fmri_data('voxelwise_correlations_OASIS_valence_amygdala.nii');

% Fisher transform correlation maps 
tdat.dat = atanh(tdat.dat);

% load niftis with correlation maps from IAPS for valence
dat=fmri_data('voxelwise_correlations_IAPS_valence_amygdala.nii');

% average Fisher transformed IAPS and OASIS correlation maps 
dat.dat = (atanh(dat.dat)+tdat.dat)/2;

% do a ttest on the averaged correlation maps
t = ttest(dat);

orthviews(t);

% threshold and save
table(region(threshold(t,.05,'UNC')));
t.fullpath = 'voxelwise_correlations_valence_amygdala.nii';
t.write;

%% interaction
% load niftis with correlation maps from OASIS for interaction
tdat=fmri_data('voxelwise_correlations_OASIS_valence_arousal_interaction_amygdala.nii');

% Fisher transform correlation maps 
tdat.dat = atanh(tdat.dat);

% load niftis with correlation maps from IAPS for interaction
dat=fmri_data('voxelwise_correlations_IAPS_valence_arousal_interaction_amygdala.nii');

% average Fisher transformed IAPS and OASIS correlation maps 
dat.dat = (atanh(dat.dat)+tdat.dat)/2;

% do a ttest on the averaged correlation maps
t = ttest(dat);

orthviews(t);

% threshold and save
table(region(threshold(t,.05,'UNC')));
t.fullpath = 'voxelwise_correlations_valence_arousal_interaction_amygdala.nii';
t.write;