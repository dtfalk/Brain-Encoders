% This script makes violin plots of amygdala encoding model performance by subregion

addpath(genpath('Github'))

load('/home/gjang/amygdala_subregions_atanh.mat')

figure;

barplot_columns(avg_atanh_subregion)
set(gca,'XTickLabels',{'LB' 'SF' 'CM' 'AStr'})
xlabel 'Amygdala Subregions'
ylabel 'Performance (Fisher''s Z)'


