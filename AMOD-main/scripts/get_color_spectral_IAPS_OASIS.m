% This script gets the color (RGB) and spatial frequency (spectral power) values for IAPS and OASIS images

% get IAPS or OASIS images
imgs = dir(['/home/data/eccolab/OASIS/Images/*.jpg']);
%imgs= dir(['/home/data/eccolab/IAPS/*.jpg']);

c = 0;

[x,y] = meshgrid(1:227,1:227);
f = sqrt((x-227/2).^2+(y-227/2).^2);

% define frequencies
low_freq = f < 6;
high_freq = f > 24;

for i = 1:length(imgs)
    
    c = c + 1;

    newImage = ([imgs(i).folder '/' imgs(i).name]);

    img = readAndPreprocessImage(newImage);

    r(c,:) = imhist(img(:,:,1));
    g(c,:) = imhist(img(:,:,2));
    b(c,:) = imhist(img(:,:,3));

    psd = abs(fftshift(fft2(img))).^2;
    spatial_power(c,:) = single(psd(:));

    high_f(c) = mean(psd(high_freq));
    low_f(c) = mean(psd(low_freq));

end

median_red = median(r');
median_green = median(g');
median_blue = median(b');

save(['compare_images_OASIS_color_red.mat'],'median_red');
save(['compare_images_OASIS_color_green.mat'],'median_green');
save(['compare_images_OASIS_color_blue.mat'],'median_blue');
save(['compare_images_OASIS_spatialFreq_high.mat'],'high_f');
save(['compare_images_OASIS_spatialFreq_low.mat'],'low_f');