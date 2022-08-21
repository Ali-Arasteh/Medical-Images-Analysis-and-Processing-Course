%% Question 1
img = rgb2gray(imread('histogram.jpeg'));
%% a)
hist_img = hist(reshape(img, 1, []), 0:255);
hist_img = hist_img/(size(img, 1)*size(img, 2));
map = zeros(256, 1);
sum = 0;
for i=1:256
    sum = sum+hist_img(i);
    map(i) = round(255*sum);
end
eq_img_f = img;
for i=1:size(img, 1)
   for j=1:size(img, 2)
       eq_img_f(i, j) = map(img(i, j)+1);
   end
end
eq_hist_img = zeros(256, 1);
for i=1:256
    eq_hist_img(map(i)+1) = eq_hist_img(map(i)+1)+hist_img(i);
end
figure();
subplot(3, 2, [1, 3]);
imshow(img);
title('image');
subplot(3, 2, [2, 4]);
imshow(eq_img_f);
title('equalized image');
subplot(3, 2, 5);
bar(0:255, hist_img);
title('histogram of image');
subplot(3, 2, 6);
bar(0:255, eq_hist_img);
title('equalized histogram of image');
%% b)
hist_img_f = hist_calculator(img);
eq_hist_img_f = hist_calculator(eq_img_f);
figure();
subplot(3, 2, [1, 3]);
imshow(img);
title('image');
subplot(3, 2, [2, 4]);
imshow(eq_img_f);
title('equalized image');
subplot(3, 2, 5);
bar(0:255, hist_img_f);
title('histogram of image - using function');
subplot(3, 2, 6);
bar(0:255, eq_hist_img_f);
title('equalized histogram of image - using function');
%% c)
hist_img_f = hist_calculator(img);
eq_hist_img_f = hist_calculator(eq_img_f);
[lo_eq_img_f, lo_eq_hist_img_f] = local_hist_calculator(img);
figure();
subplot(3, 3, [1, 4]);
imshow(img);
title('image');
subplot(3, 3, [2, 5]);
imshow(eq_img_f);
title('equalized image');
subplot(3, 3, [3, 6]);
imshow(lo_eq_img_f);
title('local equalized image');
subplot(3, 3, 7);
bar(0:255, hist_img_f);
title('histogram of image - using function');
subplot(3, 3, 8);
bar(0:255, eq_hist_img_f);
title('equalized histogram of image - using function');
subplot(3, 3, 9);
bar(0:255, lo_eq_hist_img_f);
title('local equalized histogram of image - using function');
%% Question 2
img = rgb2gray(imread('brainMRI.png'));
%% a)
noisy_img = img;
mid_x = round(size(img, 1)/2);
mid_y = round(size(img, 2)/2);
noisy_img(1:mid_x, 1:mid_y) = imnoise(img(1:mid_x, 1:mid_y), 'salt & pepper', 0.05);
noisy_img(mid_x+1:end, 1:mid_y) = imnoise(img(mid_x+1:end, 1:mid_y), 'gaussian', 0, 0.05);
noisy_img(mid_x+1:end, mid_y+1:end) = imnoise(imnoise(img(mid_x+1:end, mid_y+1:end), 'gaussian', 0, 0.05), 'salt & pepper', 0.05);
figure()
imshow(noisy_img)
title('noisy image')
%% b)
average_filtered_img = imfilter(noisy_img, fspecial('average', 5));
median_filtered_img = medfilt2(noisy_img,[5 5]);
gaussian_filtered_img = imfilter(noisy_img, fspecial('gaussian', 5, 1));
figure()
subplot(2, 2, 1)
imshow(noisy_img)
title('noisy image')
subplot(2, 2, 2)
imshow(average_filtered_img)
title('average filtered image')
subplot(2, 2, 3)
imshow(median_filtered_img)
title('median filtered image')
subplot(2, 2, 4)
imshow(gaussian_filtered_img)
title('gaussian filtered image')
%% C 
gaussian_median_filtered_img = imfilter(medfilt2(noisy_img, [5 5]), fspecial('gaussian', 5, 1));
median_gaussian_filtered_img = medfilt2(imfilter(noisy_img, fspecial('gaussian', 5, 1)), [5 5]);
figure()
subplot(1, 3, 1)
imshow(noisy_img)
title('noisy image')
subplot(1, 3, 2)
imshow(gaussian_median_filtered_img)
title('gaussian of median filtered image')
subplot(1, 3, 3)
imshow(median_gaussian_filtered_img)
title('median of gaussian filtered image')
%% d)
wiener_filtered_image = wiener2(noisy_img, [5 5], 0.1);
figure()
subplot(1, 2, 1)
imshow(noisy_img)
title('noisy image')
subplot(1, 2, 2)
imshow(wiener_filtered_image)
title('Wiener filtered image')
%% Question 3
img = rgb2gray(imread('wall.jpeg'));
%% a)
figure();
filtered_img = imfilter(img, [1, -1]);
imshow(filtered_img);
title('filtered image with kernel [1, -1]');
figure();
filtered_img = imfilter(img, [1, 0, -1]);
imshow(filtered_img);
title('filtered image with kernel [1, 0, -1]');
figure();
filtered_img = imfilter(img, [1, 0, -1]');
imshow(filtered_img);
title('filtered image with kernel transpose([1, 0, -1])');
figure();
filtered_img = imfilter(img, [1, -2, 1]);
imshow(filtered_img);
title('filtered image with kernel [1, -2, 1]');
figure();
filtered_img = imfilter(img, [1, -2, 1]');
imshow(filtered_img);
title('filtered image with kernel transpose([1, -2, 1])');
%% b)
figure();
filtered_img = edge(img,'Sobel');
imshow(filtered_img);
title('filtered image with Sobel method');
figure();
filtered_img = edge(img,'Canny', 0.4);
imshow(filtered_img);
title('filtered image with Canny method');
%% c)
figure();
filtered_img = imfilter(img, [-1, -1, -1; -1, 8, -1; -1, -1, -1]);
imshow(filtered_img);
title('filtered image with laplacian kernel');
%% d)
figure();
filtered_img = medfilt2(imfilter(img, [0, 0, -1, 0, 0; 0, -1, -2, -1, 0; -1, -2, 16, -2, -1; 0, -1, -2, -1, 0;  0, 0, -1, 0, 0]), [5, 5]);
imshow(filtered_img);
title('filtered image with LoG kernel');
%% Question 4
%% a)
foot_img = rgb2gray(imread('foot.jpg'));
hand_img = rgb2gray(imread('hand.jpg'));
foot_f = fftshift(fft2(foot_img));
hand_f = fftshift(fft2(hand_img));
figure()
subplot(1, 2, 1)
imshow(abs(foot_f)/max(abs(foot_f), [], 'all'))
title('fft of foot image')
subplot(1, 2, 2)
imshow(abs(hand_f)/max(abs(hand_f), [], 'all'))
title('fft of hand image')
%% b)
figure()
subplot(1, 2, 1)
imshow(log10(abs(foot_f)+1)/max(log10(abs(foot_f)+1), [], 'all'))
title('adjusted fft of foot image')
subplot(1, 2, 2)
imshow(log10(abs(hand_f)+1)/max(log10(abs(hand_f)+1), [], 'all'))
title('adjusted fft of hand image')
%% c)
foot_f_magnitude = abs(foot_f);
hand_f_magnitude = abs(hand_f);
foot_f_angle = angle(foot_f);
hand_f_angle = angle(hand_f);
foot_hand = ifft2(foot_f_magnitude.*(cos(hand_f_angle)+1j*sin(hand_f_angle)));
hand_foot = ifft2(hand_f_magnitude.*(cos(foot_f_angle)+1j*sin(foot_f_angle)));
figure()
subplot(1,2,1)
imshow(abs(foot_hand)/max(abs(foot_hand), [], 'all'))
title('foot magnitude and hand angle')
subplot(1,2,2)
imshow(abs(hand_foot)/max(abs(hand_foot), [], 'all'))
title('hand magnitude and foot angle')
%% Question 5
%% a)
data = zeros(360*360, 6);
figure()
for i=1:6
  img = rgb2gray(imread(['river/river', num2str(i), '.png']));
  subplot(2, 3, i)
  imshow(img)
  title(['river ', num2str(i)])
  data(:, i) = reshape(img, 1, []);
end
cov_matrix = cov(data);
[V, D] = eig(cov_matrix);
whitened_data = (data-mean(data))*V;
figure()
for i=1:6
    subplot(2,3,i)
    imshow(reshape(whitened_data(:, i), 360, 360)/max(whitened_data(:, i)))
    title(['whitened river ', num2str(i)])
end
cov_whitened_data = cov(whitened_data);
%% b)
whitened_data = whitened_data(:, 6:-1:1);
V = V(:, 6:-1:1);
D = D(6:-1:1, 6:-1:1);
clipped_whitened_data = whitened_data(:, 1:2);
clipped_V = V(:, 1:2)';
compressed_data = clipped_whitened_data*clipped_V+mean(data);
figure()
for i=1:6
    subplot(2, 3, i)
    imshow(reshape(compressed_data(:, i), 360, 360)/max(compressed_data(:, i)))
    title(['clipped river ', num2str(i)])
end
%% functions
function hist_img = hist_calculator(img)
hist_img = zeros(256, 1);
for i=1:size(img, 1)
   for j=1:size(img, 2)
       hist_img(img(i, j)+1) = hist_img(img(i, j)+1)+1;
   end
end
hist_img = hist_img/(size(img, 1)*size(img, 2));
end
function [lo_eq_img, lo_eq_hist_img] = local_hist_calculator(img)
lo_eq_img = img;
temp = 64;
for i=1:size(img, 1)
   for j=1:size(img, 2)
       temp_min_x = max(1, i-temp);
       temp_max_x = min(i+temp, size(img, 1));
       temp_min_y = max(1, j-temp);
       temp_max_y = min(j+temp, size(img, 2));
       lo_eq_img(i, j) = round(length(find(img(temp_min_x:temp_max_x, temp_min_y:temp_max_y) <= img(i, j)))/((temp_max_x-temp_min_x+1)*(temp_max_y-temp_min_y+1))*255);
   end
end
lo_eq_hist_img = zeros(256, 1);
for i=1:size(lo_eq_img, 1)
   for j=1:size(lo_eq_img, 2)
       lo_eq_hist_img(lo_eq_img(i, j)+1) = lo_eq_hist_img(lo_eq_img(i, j)+1)+1;
   end
end
lo_eq_hist_img = lo_eq_hist_img/(size(lo_eq_img, 1)*size(lo_eq_img, 2));
end