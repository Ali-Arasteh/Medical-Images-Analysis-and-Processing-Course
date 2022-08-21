%% Question 1
%% 1-1)
for i = 1:4
    mri(:, :, i)  = double(imread(['MRI', num2str(i), '.png']))/255;
end
figure();
imshow(mri(:, :, 1:3))
title('RGB image from images MRI1, MRI2, MRI3')
figure();
imshow(mri(:, :, 2:4));
title('RGB image from images MRI2, MRI3, MRI4')
%% 1-2)
[m, n, p] = size(mri);
cluster_num = 4;
fuzzy_coefficient = [1.1, 2, 5];
for fc = fuzzy_coefficient
    options = [fc NaN NaN 0];
    [~, U, obj_f] = fcm(reshape(mri(:, :, :), [], p), cluster_num, options);
    fprintf([num2str(length(obj_f)), ' iterations for fuzzy coefficient ', num2str(fc), ', loss = ', num2str(obj_f(end)), '\n']);
    figure();
    subplot(2, 2, 1);
    imshow(reshape(U(1, :), m, n));
    title('Probability map of cluster 1');
    subplot(2, 2, 2);
    imshow(reshape(U(2, :), m, n));
    title('Probability map of cluster 2');
    subplot(2, 2, 3);
    imshow(reshape(U(3, :), m, n));
    title('Probability map of cluster 3');
    subplot(2, 2, 4);
    imshow(reshape(U(4, :), m, n));
    title('Probability map of cluster 4');
    sgtitle(['Probability maps for fuzzy coefficient ', num2str(fc)]);
end
%% 1-3)
cluster_num = [2, 3, 4, 5];
for cn = cluster_num
    figure();
    indices = kmeans(reshape(mri(:, :, :), [], p), cn);
    imshow(labeloverlay(zeros(m, n, 'uint8'), reshape(indices, m, n), 'Colormap', colormap(parula(cn))));
    title(['Kmeans with ', num2str(cn), ' clusters']);
end
cluster_num = 4;
indices = kmeans(reshape(mri(:, :, :), [], p), cluster_num);
init_U = zeros(cluster_num, length(indices));
for i = 1:length(indices)
	init_U(indices(i), i) = 1;
end
fuzzy_coefficient = [1.1, 2, 5];
for fc = fuzzy_coefficient
    options = [fc, NaN, NaN, 0];
    [~, U, obj_f] = adjusted_fcm(reshape(mri(:, :, :), [], p), cluster_num, options, init_U);
    fprintf([num2str(length(obj_f)), ' iterations for fuzzy coefficient ', num2str(fc), ', loss = ', num2str(obj_f(end)), '\n']);
    figure();
    subplot(2, 2, 1);
    imshow(reshape(U(1, :), m, n));
    title('Probability map of cluster 1');
    subplot(2, 2, 2);
    imshow(reshape(U(2, :), m, n));
    title('Probability map of cluster 2');
    subplot(2, 2, 3);
    imshow(reshape(U(3, :), m, n));
    title('Probability map of cluster 3');
    subplot(2, 2, 4);
    imshow(reshape(U(4, :), m, n));
    title('Probability map of cluster 4');
    sgtitle(['Probability maps for fuzzy coefficient ', num2str(fc)]);
end
%% 1-4)
GMM_model = fitgmdist(reshape(mri(:, :, :), [], p), cluster_num, 'RegularizationValue', 0.005);
indices = cluster(GMM_model, reshape(mri(:, :, :), [], p));
imshow(labeloverlay(zeros(m, n, 'uint8'), reshape(indices, m, n), 'Colormap', colormap(parula(cluster_num))));
title(['GMM with ', num2str(cluster_num), ' clusters']);
%% 1-5)
cluster_num = 4;
indices = kmeans(reshape(mri(:, :, :), [], p), cluster_num);
init_U = zeros(cluster_num, length(indices));
for i = 1:length(indices)
	init_U(indices(i), i) = 1;
end
fuzzy_coefficient = [1.1, 2, 5];
for fc = fuzzy_coefficient
    options = [fc, NaN, NaN, 0];
    [~, U, obj_f] = adjusted_fcm(reshape(mri(:, :, :), [], p), cluster_num, options, init_U);
    figure();
    imshow(reshape((max(U, [], 1) <= max(1.25/cluster_num, 1/fc)), m, n));
    sgtitle(['Partial volumes for fuzzy coefficient ', num2str(fc)]);
end
%% Question 2
%% 2-1)
nevus = imread('nevus_gray.jpg');
melanoma = imread('melanoma_gray.jpg');
% GVF
addpath('./snake_demo/snake');
[u, v] = GVF(nevus, 0.05, 1000);
figure();
quiver(flip(u, 1), flip(v, 1));
axis('off');
title('nevus');
figure();
imshow(labeloverlay(nevus, (sqrt(u.^2+v.^2)/max(sqrt(u.^2+v.^2), [], 'all')) > 0.5));
title('GVF output on nevus');
[u, v] = GVF(melanoma, 0.05, 1000);
figure();
quiver(flip(u, 1), flip(v, 1));
axis('off');
title('melanoma');
figure();
imshow(labeloverlay(melanoma, (sqrt(u.^2+v.^2)/max(sqrt(u.^2+v.^2), [], 'all')) > 0.5));
title('GVF output on melanoma');
% basic snake
addpath('./activeContoursSnakesDemo/activeContoursDemo');
snk();
%% 2-2)
% GVF
img = imread('MRI1.jpg');
[u, v] = GVF(img, 0.05, 1000);
figure();
quiver(flip(u, 1), flip(v, 1));
axis('off');
title('MRI1');
figure();
imshow(labeloverlay(img, (sqrt(u.^2+v.^2)/max(sqrt(u.^2+v.^2), [], 'all')) > 0.5));
title('GVF output on MRI1');
% basic snake
snk();
%% Question 3
%% 3-1)
nevus = imread('nevus_gray.jpg');
melanoma = imread('melanoma_gray.jpg');
addpath('./Chan-Vese');
% mask
mask = roipoly(nevus);
chenvese(nevus, mask, 500, 0.1, 'chan');
mask = roipoly(melanoma);
chenvese(melanoma, mask, 500, 0.1, 'chan');
% point
imshow(nevus); 
[y, x] = ginput(1);
mask = zeros(size(nevus));
mask(max(x-4, 1):min(x+4, size(nevus, 1)),max(y-4, 1):min((y+4), size(nevus, 2))) = 1;
chenvese(nevus, mask, 500, 0.1, 'chan');
imshow(melanoma); 
[y, x] = ginput(1);
mask = zeros(size(melanoma));
mask(max(x-4, 1):min(x+4, size(melanoma, 1)),max(y-4, 1):min((y+4), size(melanoma, 2))) = 1;
chenvese(melanoma, mask, 500, 0.1, 'chan');
%% 3-2)
img = imread('MRI3.png');
% mask
mask = roipoly(img);
chenvese(img, mask, 500, 0.1, 'chan');
% point
imshow(img); 
[y, x] = ginput(1);
mask = zeros(size(img));
mask(max(x-4, 1):min(x+4, size(img, 1)),max(y-4, 1):min((y+4), size(img, 2))) = 1;
chenvese(img, mask, 500, 0.1, 'chan');
% auto segmentation
median_img = medfilt2(img, [25 25]);
[x, y] = find(median_img == max(median_img, [], 'all'), 1);
mask = zeros(size(img));
mask(max(x-4, 1):min(x+4, size(img, 1)),max(y-4, 1):min((y+4), size(img, 2))) = 1;
chenvese(img, mask, 500, 0.1, 'chan');