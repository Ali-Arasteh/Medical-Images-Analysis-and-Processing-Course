%% Question 1
addpath('BCFCM');
%% Part a)
% in report
%% Part b)
img = double(imread('test_biasfield_noise.png'))/255;
num_cluster = 2;
Options.epsilon = 10e-6;
Options.alpha = 0.85;
Options.sigma = 1;
Options.p = 2;
[bf, u] = BCFCM2D(img, [0, 1]', Options);
figure();
subplot(1, 3, 1);
imshow(img);
title('Original Image');
subplot(1, 3, 2);
imshow(bf);
title('Bias Field');
subplot(1, 3, 3);
imshow(img - bf);
title('Corrected Image');
figure();
for i = 1:num_cluster
    subplot(1, num_cluster, i);
    imshow(u(:, :, i));
    title(['u_', num2str(i)]);
end
%% Question 2
addpath('Snake_GVF/');
addpath('Snake_GVF/Snake_GVF');
%% Part a)
% in report
%% Part b)
% in report
%% Part c)
% in report
%% Part d)
I = double(imread('example.png'))/255;
[~, y, x] = roipoly(I);
x(end) = [];
y(end) = [];
%%
Options = [];
Options.Verbose = 1;
% Options.nPoints = 100;
% Options.Gamma = 1;
Options.Iterations = 200;
% Options.Sigma1 = 10;
% Options.Wline = 0.04;
% Options.Wedge = 2;
% Options.Wterm = 0.01;
% Options.Sigma2 = 20;
% Options.Mu = 0.2;
Options.GIterations = 50;
% Options.Sigma3 = 1;
Options.Alpha = 0.4;
Options.Beta = 2;
Options.Delta = 0.4;
Options.Kappa = 4;
[O, J] = Snake2D(I, [x, y], Options);
J = double(J);
Irgb = [];
Irgb(:,:,1) = I;
Irgb(:,:,2) = I;
Irgb(:,:,3) = J;
figure();
imshow(Irgb);