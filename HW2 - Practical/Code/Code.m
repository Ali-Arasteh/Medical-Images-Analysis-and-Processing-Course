%% Question 1
img1 = ~imbinarize(rgb2gray(imread('melanome1.jpg')), 0.25);
img2 = ~imbinarize(rgb2gray(imread('melanome2.jpg')), 0.25);
img3 = ~imbinarize(rgb2gray(imread('melanome3.jpg')), 0.25);
img4 = ~imbinarize(rgb2gray(imread('melanome4.jpg')), 0.25);
figure();
subplot(2, 2, 1);
imshow(img1);
title('meloanome 1');
subplot(2, 2, 2);
imshow(img2);
title('meloanome 2');
subplot(2, 2, 3);
imshow(img3);
title('meloanome 3');
subplot(2, 2, 4);
imshow(img4);
title('meloanome 4');
%% a)
connected_img1 = imclose(img1, strel('rectangle',[5, 5]));
figure();
subplot(1, 2, 1);
imshow(img1);
subplot(1, 2, 2);
imshow(connected_img1);
sgtitle('Connecting seperated components using closing operation');
n_comp1 = bwconncomp(connected_img1).NumObjects;
%% b)
denoised_img2 = imopen(img2, strel('rectangle',[3, 3]));
figure();
subplot(1, 2, 1);
imshow(img2);
subplot(1, 2, 2);
imshow(denoised_img2);
sgtitle('Denoising melanoma using opening operation');
boundary2 = denoised_img2-imerode(denoised_img2, strel('rectangle',[3, 3]));
figure();
imshow(boundary2);
title('boundary of melanoma');
%% c)
connected_img3 = imclose(img3, strel('rectangle',[3, 3]));
filled_img3 = imfill(connected_img3,'holes');
figure();
imshow(filled_img3);
title('filling melanoma using imfill');
%% d)
seperated_img4 = imopen(img4, strel('rectangle',[21, 21]));
figure();
subplot(1, 2, 1);
imshow(img4);
subplot(1, 2, 2);
imshow(seperated_img4);
sgtitle('seperating two melanomas using opening operation');
%% e)
components = connected_component(seperated_img4);
for i=1:size(components, 1)
    figure();
    imshow(squeeze(components(i, :, :)));
    title(['connected component ', num2str(i)]);
end
%% Question 2
%% a)
brain_img = imread('brain.jpg');
figure();
imshow(brain_img);
title('brain');
hist_img = hist(reshape(brain_img, 1, []), 0:255);
norm_hist_img = hist_img/sum(hist_img);
figure();
bar(0:255, norm_hist_img);
title('normalized histogram of brain');
brain_mean = sum((0:255).*norm_hist_img);
brain_variance = sum((((0:255)-brain_mean).^2).*norm_hist_img);
brain_uniformity = sum(norm_hist_img.^2);
brain_entropy = -sum(norm_hist_img.*log2(norm_hist_img+1e-12));
%% b)
sample_img = imread('sample.png');
figure();
imshow(sample_img);
title('sample');
hist_img = hist(reshape(sample_img, 1, []), 0:255);
norm_hist_img = hist_img/sum(hist_img);
figure();
bar(0:255, norm_hist_img);
title('normalized histogram of sample');
sample_average = sum((0:255).*norm_hist_img);
sample_variance = sum((((0:255)-sample_average).^2).*norm_hist_img);
sample_uniformity = sum(norm_hist_img.^2);
sample_entropy = -sum(norm_hist_img.*log2(norm_hist_img+1e-10));
%% c)
brain_GLCM_256 = GLCM(brain_img, 256, [0, 1]);
figure();
imshow(log(brain_GLCM_256+1)/max(log(brain_GLCM_256+1), [], 'all'));
title('brain GLCM 256 for offset = [0, 1]');
sample_GLCM_256 = GLCM(sample_img, 256, [0, 1]);
figure();
imshow(log(sample_GLCM_256+1)/max(log(sample_GLCM_256+1), [], 'all'));
title('sample GLCM 256 for offset = [0, 1]');
%% d)
norm_brain_GLCM_256 = brain_GLCM_256/sum(brain_GLCM_256, 'all');
norm_sample_GLCM_256 = sample_GLCM_256/sum(sample_GLCM_256, 'all');
brain_contrast_256 = 0;
sample_contrast_256 = 0;
for i=1:size(brain_GLCM_256, 1)
    for j=1:size(brain_GLCM_256, 1)
        brain_contrast_256 = brain_contrast_256+(i-j)^2*norm_brain_GLCM_256(i, j);
        sample_contrast_256 = sample_contrast_256+(i-j)^2*norm_sample_GLCM_256(i, j);
    end
end
brain_uniformity_256 = 0;
sample_uniformity_256 = 0;
for i=1:size(brain_GLCM_256, 1)
    for j=1:size(brain_GLCM_256, 1)
        brain_uniformity_256 = brain_uniformity_256+norm_brain_GLCM_256(i, j)^2;
        sample_uniformity_256 = sample_uniformity_256+norm_sample_GLCM_256(i, j)^2;
    end
end
brain_homogeneity_256 = 0;
sample_homogeneity_256 = 0;
for i=1:size(brain_GLCM_256, 1)
    for j=1:size(brain_GLCM_256, 1)
        brain_homogeneity_256 = brain_homogeneity_256+norm_brain_GLCM_256(i, j)/(1+abs(i-j));
        sample_homogeneity_256 = sample_homogeneity_256+norm_sample_GLCM_256(i, j)/(1+abs(i-j));
    end
end
brain_entropy_256 = 0;
sample_entropy_256 = 0;
for i=1:size(brain_GLCM_256, 1)
    for j=1:size(brain_GLCM_256, 1)
        brain_entropy_256 = brain_entropy_256-norm_brain_GLCM_256(i, j)*log2(norm_brain_GLCM_256(i, j)+1e-10);
        sample_entropy_256 = sample_entropy_256-norm_sample_GLCM_256(i, j)*log2(norm_sample_GLCM_256(i, j)+1e-10);
    end
end
%% e)
brain_GLCM_128 = brain_GLCM_256(1:2:255, 1:2:255)+brain_GLCM_256(1:2:255, 2:2:256)+brain_GLCM_256(2:2:256, 1:2:255)+brain_GLCM_256(2:2:256, 2:2:256);
figure();
imshow(log(brain_GLCM_128+1)/max(log(brain_GLCM_128+1), [], 'all'));
title('brain GLCM 128 for offset = [0, 1]');
sample_GLCM_128 = sample_GLCM_256(1:2:255, 1:2:255)+sample_GLCM_256(1:2:255, 2:2:256)+sample_GLCM_256(2:2:256, 1:2:255)+sample_GLCM_256(2:2:256, 2:2:256);
figure();
imshow(log(sample_GLCM_128+1)/max(log(sample_GLCM_128+1), [], 'all'));
title('sample GLCM 128 for offset = [0, 1]');
norm_brain_GLCM_128 = brain_GLCM_128/sum(brain_GLCM_128, 'all');
norm_sample_GLCM_128 = sample_GLCM_128/sum(sample_GLCM_128, 'all');
brain_contrast_128 = 0;
sample_contrast_128 = 0;
for i=1:size(brain_GLCM_128, 1)
    for j=1:size(brain_GLCM_128, 1)
        brain_contrast_128 = brain_contrast_128+(i-j)^2*norm_brain_GLCM_128(i, j);
        sample_contrast_128 = sample_contrast_128+(i-j)^2*norm_sample_GLCM_128(i, j);
    end
end
brain_uniformity_128 = 0;
sample_uniformity_128 = 0;
for i=1:size(brain_GLCM_128, 1)
    for j=1:size(brain_GLCM_128, 1)
        brain_uniformity_128 = brain_uniformity_128+norm_brain_GLCM_128(i, j)^2;
        sample_uniformity_128 = sample_uniformity_128+norm_sample_GLCM_128(i, j)^2;
    end
end
brain_homogeneity_128 = 0;
sample_homogeneity_128 = 0;
for i=1:size(brain_GLCM_128, 1)
    for j=1:size(brain_GLCM_128, 1)
        brain_homogeneity_128 = brain_homogeneity_128+norm_brain_GLCM_128(i, j)/(1+abs(i-j));
        sample_homogeneity_128 = sample_homogeneity_128+norm_sample_GLCM_128(i, j)/(1+abs(i-j));
    end
end
brain_entropy_128 = 0;
sample_entropy_128 = 0;
for i=1:size(brain_GLCM_128, 1)
    for j=1:size(brain_GLCM_128, 1)
        brain_entropy_128 = brain_entropy_128-norm_brain_GLCM_128(i, j)*log2(norm_brain_GLCM_128(i, j)+1e-10);
        sample_entropy_128 = sample_entropy_128-norm_sample_GLCM_128(i, j)*log2(norm_sample_GLCM_128(i, j)+1e-10);
    end
end
%% f)
% Anwsered in report
%% Question 3
%% a)
% Anwsered in report
%% b)
boat_img = imread('boat.png');
lowpass = 1/(4*sqrt(2))*[1+sqrt(3), 3+sqrt(3), 3-sqrt(3), 1-sqrt(3)];
highpass = 1/(4*sqrt(2))*[1-sqrt(3), -3+sqrt(3), 3+sqrt(3), -1-sqrt(3)];
row_low_boat = downsample(conv2(1, lowpass, boat_img, 'same')', 2, 1)';
A_boat = downsample(conv2(lowpass, 1, row_low_boat, 'same'), 2, 1);
D_horizental_boat = downsample(conv2(highpass, 1, row_low_boat, 'same'), 2, 1);
row_high_boat = downsample(conv2(1, highpass, boat_img, 'same')', 2, 1)';
D_vertical_boat = downsample(conv2(lowpass, 1, row_high_boat, 'same'), 2, 1);
D_diagonal_boat = downsample(conv2(highpass, 1, row_high_boat, 'same'), 2, 1);
figure();
subplot(2, 2, 1);
imshow(A_boat/max(A_boat, [],'all'));
title('A');
subplot(2, 2, 2);
imshow(D_horizental_boat/max(D_horizental_boat, [],'all'));
title('D horizental');
subplot(2, 2, 3);
imshow(D_vertical_boat/max(D_vertical_boat, [],'all'));
title('D vertical');
subplot(2, 2, 4);
imshow(D_diagonal_boat/max(D_diagonal_boat, [],'all'));
title('D diagonal');
%% c)
re_lowpass = 1/(4*sqrt(2))*[3-sqrt(3), 3+sqrt(3), 1+sqrt(3), 1-sqrt(3)];
re_highpass = 1/(4*sqrt(2))*[1-sqrt(3), -1-sqrt(3), 3+sqrt(3), -3+sqrt(3)];
re_A_boat = conv2(re_lowpass, 1, upsample(A_boat, 2, 1), 'same');
re_D_horizental_boat = conv2(re_highpass, 1, upsample(D_horizental_boat, 2, 1), 'same');
re_D_vertical_boat = conv2(re_lowpass, 1, upsample(D_vertical_boat, 2, 1), 'same');
re_D_diagonal_boat = conv2(re_highpass, 1, upsample(D_diagonal_boat, 2, 1), 'same');
re_boat_img = conv2(1, re_lowpass, upsample((re_A_boat+re_D_horizental_boat)', 2, 1)', 'same')+conv2(1, re_highpass, upsample((re_D_vertical_boat+re_D_diagonal_boat)', 2, 1)', 'same');
re_boat_img = (re_boat_img-min(re_boat_img, [], 'all'))/(max(re_boat_img, [], 'all')-min(re_boat_img, [], 'all'))*255;
figure();
imshow(re_boat_img/max(re_boat_img, [], 'all'));
title('reconstructed boat image');
rmse = sqrt(mean((double(boat_img)-re_boat_img).^2, 'all'));
%% d)
percent = 0.95;
temp = sort(reshape([A_boat, D_horizental_boat, D_vertical_boat, D_diagonal_boat], 1, []), 'descend');
threshold = temp(round(percent*length(temp)));
compressed_A_boat = A_boat.*(A_boat > threshold);
compressed_D_horizental_boat = D_horizental_boat.*(D_horizental_boat > threshold);
compressed_D_vertical_boat = D_vertical_boat.*(D_vertical_boat > threshold);
compressed_D_diagonal_boat = D_diagonal_boat.*(D_diagonal_boat > threshold);
compressed_re_A_boat = conv2(re_lowpass, 1, upsample(compressed_A_boat, 2, 1), 'same');
compressed_re_D_horizental_boat = conv2(re_highpass, 1, upsample(compressed_D_horizental_boat, 2, 1), 'same');
compressed_re_D_vertical_boat = conv2(re_lowpass, 1, upsample(compressed_D_vertical_boat, 2, 1), 'same');
compressed_re_D_diagonal_boat = conv2(re_highpass, 1, upsample(compressed_D_diagonal_boat, 2, 1), 'same');
compressed_re_boat_img = conv2(1, re_lowpass, upsample((compressed_re_A_boat+compressed_re_D_horizental_boat)', 2, 1)', 'same')+conv2(1, re_highpass, upsample((compressed_re_D_vertical_boat+compressed_re_D_diagonal_boat)', 2, 1)', 'same');
compressed_re_boat_img = (compressed_re_boat_img-min(compressed_re_boat_img, [], 'all'))/(max(compressed_re_boat_img, [], 'all')-min(compressed_re_boat_img, [], 'all'))*255;
figure();
imshow(compressed_re_boat_img/max(compressed_re_boat_img, [], 'all'));
title([num2str(percent*100),'% compressed reconstructed boat image']);
compressed_rmse = sqrt(mean((double(boat_img)-compressed_re_boat_img).^2, 'all'));
%% e)
uint8_compressed_re_boat_img = uint8(round(compressed_re_boat_img/max(compressed_re_boat_img, [], 'all')*255));
imwrite(uint8_compressed_re_boat_img, ['compressed boat ', num2str(round(percent*100)), ' percent.png']);
%% Question 4
%% a)
covid_img = rgb2gray(imread('covid.png'));
[LoD, HiD] = wfilters('haar', 'd');
[cA, cH, cV, cD] = dwt2(covid_img, LoD, HiD);
figure();
subplot(2, 2, 1);
imshow(cA/max(cA, [],'all'));
title('A');
subplot(2, 2, 2);
imshow(cH/max(cH, [],'all'));
title('D horizental');
subplot(2, 2, 3);
imshow(cV/max(cV, [],'all'));
title('D vertical');
subplot(2, 2, 4);
imshow(cD/max(cD, [],'all'));
title('D diagonal');
cH = cH.*(cH > 40);
cV = cV.*(cV > 40);
cD = cD.*(cD > 20);
figure();
subplot(2, 2, 1);
imshow(cA/max(cA, [],'all'));
title('A');
subplot(2, 2, 2);
imshow(cH/max(cH, [],'all'));
title('D horizental');
subplot(2, 2, 3);
imshow(cV/max(cV, [],'all'));
title('D vertical');
subplot(2, 2, 4);
imshow(cD/max(cD, [],'all'));
title('D diagonal');
%% b)
[LoR, HiR] = wfilters('haar', 'r');
re_covid_img = idwt2(cA, cH, cV, cD, LoR, HiR);
figure();
imshow(re_covid_img/max(re_covid_img, [], 'all'));
title('denoised covid image using Haar wavelet');
%% c)
covid_frequendy = fft2(covid_img);
covid_frequendy = covid_frequendy.*(abs(covid_frequendy)> 2000);
re_covid_img = ifft2(covid_frequendy);
figure();
imshow(real(re_covid_img)/max(real(re_covid_img), [], 'all'));
title('denoised covid image using DFT wavelet');
%% functions
function img_components = connected_component(img)
    if islogical(img)
        img_components = zeros(1, size(img, 1), size(img, 2), 'logical');
        i = 0;
        while max(img, [], 'all')
            i = i+1;
            temp = zeros(size(img, 1), size(img, 2), 'logical');
            [x, y] = find(img == 1);
            temp(x(1), y(1)) = 1;
            while max(temp ~= (imdilate(temp, strel('rectangle',[3, 3])) & img), [], 'all')
                temp = imdilate(temp, strel('rectangle',[3, 3])) & img;
            end
            img_components(i, :, :) = temp;
            img = img-temp;
        end 
    else
        fprintf('Image must be logical');
    end
end
function output = GLCM(img, n_levels, offset)
    output = zeros(n_levels, n_levels);
    min_x = max(1, 1-offset(1));
    max_x = min(size(img, 1), size(img, 1)-offset(1));
    min_y = max(1, 1-offset(2));
    max_y = min(size(img, 2), size(img, 2)-offset(2));
    levels = ceil(linspace(-0.5, 255.5, n_levels+1));
    levels = levels(2:end);
    for i=min_x:max_x
        for j=min_y:max_y
            temp1 = find(levels > img(i, j), 1);
            temp2 = find(levels > img(i+offset(1), j+offset(2)), 1);
            output(temp1, temp2) = output(temp1, temp2)+1;
        end
    end
end