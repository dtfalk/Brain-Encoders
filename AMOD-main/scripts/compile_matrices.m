% This script compiles all subjects' amygdala activations in a matrix and does atanh conversion for normalization

addpath(genpath('Github'))

subjects = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20'}

for s = 1:length(subjects)

    dat = fmri_data(['/home/data/eccolab/OpenNeuro/ds002837/derivatives/sub-' subjects{s} '/func/sub-' subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz']);

    masked_dat = apply_mask(dat,select_atlas_subset(load_atlas('canlab2018'),{'Amy'}));

    excluded_voxels = masked_dat.removed_voxels

    load(['/home/gjang/sub-' subjects{s} '_amygdala_fc7_invert_imageFeatures_output.mat'])

    matrix(s,~excluded_voxels) = mean_diag_corr

end

save(['amygdala_fc7_invert_imageFeatures_output_compilation_matrix.mat'],'matrix')

% clean matrix by excluding empty arrays (end result: new_matrix=20x252)
new_matrix = matrix(:,~all(matrix==0))
save(['amygdala_fc7_invert_imageFeatures_output_matrix_clean.mat'],'new_matrix')

% make any values that were 0 into NaN in new_matrix
new_matrix(new_matrix==0) = NaN
save(['amygdala_fc7_invert_imageFeatures_output_matrix_nan.mat'],'new_matrix')

% do atanh conversion of the data to normalize (Fisher's Z)
atanh_matrix = atanh(new_matrix)
save(['amygdala_fc7_invert_imageFeatures_output_matrix_atanh.mat'],'atanh_matrix')