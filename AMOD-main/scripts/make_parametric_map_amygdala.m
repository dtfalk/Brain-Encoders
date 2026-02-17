% This script performs a ttest and makes a t-map of amygdala voxels, and thresholds by FDR q < .05

addpath(genpath('Github'))

subjects = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'}

% step one -- mask for amygdala voxels
for s = 1:length(subjects)

    dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-' subjects{s} '/func/sub-' subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz']);

    masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),{'Amy'}));

    excluded_voxels(s,:) = masked_dat.removed_voxels

end

% step two -- load n subj x n voxel amygdala output (atanh) for layer
load('/home/gjang/amygdala_fc7_invert_imageFeatures_output_matrix_atanh_TEST_20240702.mat')

atanh_matrix(atanh_matrix==0) = NaN

% step three -- perform a ttest on that output
[h, p, ci, stats] = ttest(atanh_matrix);

% step four -- create a statistic image object
stats_object = statistic_image;

stats_object.volInfo = masked_dat.volInfo;

stats_object.removed_voxels = all(excluded_voxels);

% step five -- assign t values to that object
stats_object.dat = stats.tstat';

% step six -- write out as nifti image
stats_object.fullpath = '/home/gjang/tstat_FisherZ_amygdala_imageFeatures_fc7.nii';

write(stats_object,'overwrite');

% thresholded by (FDR) False Discovery Rate, q < 0.05
th_stats_object = stats_object;

th_stats_object.dat(~(p<FDR(p,.05)))=NaN;

th_stats_object.fullpath = '/home/gjang/tstat_FisherZ_amygdala_imageFeatures_fc7_threshold_q05.nii';

write(th_stats_object,'overwrite');
