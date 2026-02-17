% This is a script used to generate random targets of activation for artificial stimuli generation 
% based on randomly selected voxels from the amygdala encoding models

% for 1000 iterations
for it = 1:1000

    % for 20 subjects
    for s = 1:20

        % load their amygdala encoding model
        load(['/home/data/eccolab/AMax/encoding_models/betas/compare_images/beta_sub-' num2str(s) '_amy_fc7_invert_imageFeatures.mat']);

        if s == 1
            
            % split voxels into four subregions
            split = randi(4,size(b,2),1);
        
        end
        
        % for four subregions
        for r = 1:4
            
            % average across voxels and save text file
            csvwrite(['/home/data/eccolab/AMax/encoding_models/betas/random_subregion_betas/meanbeta_region' num2str(r) '_sub-' num2str(s) '_amyFearful_fc7_invert_imageFeatures_iteration_' num2str(it) '.csv'],mean(b(2:end,split(1:size(b,2))==r)')');
        
        end
    
    end

end