%% Question 1
hand_img = double(imresize(imread('hand.jpg'), 0.2))/255;
%% a)
gray_hand_img = rgb2gray(hand_img);
noisy_hand_img = imnoise(hand_img, 'gaussian', 0.05, 0.01);
gray_noisy_hand_img = rgb2gray(noisy_hand_img);
figure();
montage([gray_hand_img, gray_noisy_hand_img]);
title('without noise and noisy grayscale images');
%% b)
[m, n] = size(gray_noisy_hand_img);
hx = 1;
weight_matrix = zeros(2*m-1, 2*n-1);
for i=1:2*m-1
    for j=1:2*n-1
       weight_matrix(i, j) = exp(-((i-m)^2+(j-n)^2)/(2*hx^2));
    end
end
CR_denoised_img = zeros(m, n);
for i=1:m
    for j=1:n
       CR_denoised_img(i, j) = sum(gray_noisy_hand_img.*weight_matrix(m-i+1:2*m-i, n-j+1:2*n-j), 'all')/sum(weight_matrix, 'all');
    end
end
figure();
montage(CR_denoised_img);
title('Classical Regression denoised image');
%% c)
[m, n] = size(gray_noisy_hand_img);
hx = 1;
hg = 0.04;
BL_denoised_img = zeros(m, n);
for i=1:m
    for j=1:n
        weight_matrix = zeros(m, n);
        for img=1:m
            for q=1:n
                weight_matrix(img, q) = exp(-((i-img)^2+(j-q)^2)/(2*hx^2))*exp(-((gray_noisy_hand_img(i)-gray_noisy_hand_img(img))^2+(gray_noisy_hand_img(j)-gray_noisy_hand_img(q))^2)/(2*hg^2));
            end
        end
        BL_denoised_img(i, j) = sum(gray_noisy_hand_img.*weight_matrix, 'all')/sum(weight_matrix, 'all');
    end
end
figure();
montage(BL_denoised_img);
title('Bilateral denoised image');
%% d)
k = 1;
hv = 0.04*(2*k+1);
NLM_denoised_img = NLM_denoising(gray_noisy_hand_img, k, hv);
figure();
montage(NLM_denoised_img);
title('NLM denoised image');
%% e)
addpath('BM3D');
BM3D_denoised_img = BM3D(gray_noisy_hand_img, 0.075);
figure();
montage(BM3D_denoised_img);
title('BM3D denoised image');
%% f)
noisy_SNR = 20*log10(norm(gray_hand_img, 'fro')/norm(gray_hand_img-gray_noisy_hand_img, 'fro'));
CR_SNR = 20*log10(norm(gray_hand_img, 'fro')/norm(gray_hand_img-CR_denoised_img, 'fro'));
BL_SNR = 20*log10(norm(gray_hand_img, 'fro')/norm(gray_hand_img-BL_denoised_img, 'fro'));
NLM_SNR = 20*log10(norm(gray_hand_img, 'fro')/norm(gray_hand_img-NLM_denoised_img, 'fro'));
BM3D_SNR = 20*log10(norm(gray_hand_img, 'fro')/norm(gray_hand_img-BM3D_denoised_img, 'fro'));
%% Question 2
%% a)
img = phantom('Modified Shepp-Logan', 700);
noisy_img = imnoise(img, 'salt & pepper', 0.03);
figure();
imshow(noisy_img);
title('noisy image');
%% b)
addpath('functions');
aniso_denoised_img = anisodiff(noisy_img, 10, 0.5, 0.1, 2);
figure();
imshow(aniso_denoised_img);
title('anisotropic denoised image');
%% C)
img_edges = imfilter(img, fspecial('laplacian', 0.2))-mean(imfilter(img, fspecial('laplacian', 0.2)), 'all');
aniso_denoised_img_edges = imfilter(aniso_denoised_img, fspecial('laplacian', 0.2))-mean(imfilter(aniso_denoised_img, fspecial('laplacian', 0.2)), 'all');
aniso_EPI = sum(img_edges.*aniso_denoised_img_edges, 'all')/(norm(img_edges, 'fro')*norm(aniso_denoised_img_edges, 'fro'));
aniso_SNR = 20*log10(norm(img, 'fro')/norm(img-aniso_denoised_img, 'fro'));
%% Question 3
%% a)
% answered in report
%% b)
addpath('functions');
TV_denoised_img = TVL1denoise(noisy_img, 1, 100);
figure();
imshow(TV_denoised_img);
title('TV denoised image');
%% C)
img_edges = imfilter(img, fspecial('laplacian', 0.2))-mean(imfilter(img, fspecial('laplacian', 0.2)), 'all');
TV_denoised_img_edges = imfilter(TV_denoised_img, fspecial('laplacian', 0.2))-mean(imfilter(TV_denoised_img, fspecial('laplacian', 0.2)), 'all');
TV_EPI = sum(img_edges.*TV_denoised_img_edges, 'all')/(norm(img_edges, 'fro')*norm(TV_denoised_img_edges, 'fro'));
TV_SNR = 20*log10(norm(img, 'fro')/norm(img-TV_denoised_img, 'fro'));
%% functions
function NLM_denoised_img = NLM_denoising(gray_noisy_img, k, hv)
    [m, n] = size(gray_noisy_img);
    padded_gray_noisy_img = padarray(gray_noisy_img,[k, k],'symmetric');
    NLM_denoised_img = zeros(m, n);
    for i=1:m
        for j=1:n
            weight_matrix = zeros(m, n);
            for p=1:m
                for q=1:n
                    weight_matrix(p, q) = exp(-norm(padded_gray_noisy_img(i:i+2*k, j:j+2*k)-padded_gray_noisy_img(p:p+2*k, q:q+2*k), 'fro')/(2*hv^2));
                end
            end
            NLM_denoised_img(i, j) = sum(gray_noisy_img.*weight_matrix, 'all')/sum(weight_matrix, 'all');
        end
        i
    end
end