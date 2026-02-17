% This script reads in movie mp4 and extracts features from every 5th frame using EmoNet

%%
load netTransfer_20cat.mat
featureLayer = 'fc7';
vids=dir(['/home/gjang/Stimuli/500_days_of_summer.mp4']);

%%
cc=0;

cc=cc+1;
clear video_*

vid=VideoReader(['/home/gjang/Stimuli/500_days_of_summer.mp4']);

numFrames = vid.NumberOfFrames;
n=numFrames;

ci=0;

for i = 1:5:n
    ci=ci+1;
    img = read(vid,i);
    img = readAndPreprocessImage (img);

    % Extract image features using the CNN
    temp_variable = activations(netTransfer,img,featureLayer);
    video_imageFeatures(ci,:)  = temp_variable(:);
    [video_maxval(ci),video_pred_label(ci)]=max(video_imageFeatures(ci,:),[],2);

end

avg_features(cc,:)=mean(video_imageFeatures);

save('500_days_of_summer_fc7_features','video_imageFeatures', '-v7.3')
