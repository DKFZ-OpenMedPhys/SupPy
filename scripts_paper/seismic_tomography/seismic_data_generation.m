
clear, clc
fprintf(1,'Starting demo_sirt:\n\n');

N = 256;        % The image is N-times-N..
s = 64;
p = 128;


% Create the test problem.
[A,b,x,s,p] = seismicwavetomo(N,s,p);

% Show the exact solution
figure(1), clf
subplot(2,3,1)
imagesc(reshape(x,N,N)), colormap gray, axis image off
c = caxis;
title('Exact phantom')

% No. of iterations.
k = [1:1000];


%%
% Perform the DROP iterations.
tic;
options = struct();
options.lbound = 0;
options.ubound = 1;
Xdrop = drop(A,b,k,[],options);
toc

%% Add noise
noiseLevel = 0.05;
noise = normrnd(0, noiseLevel .* b);
b_noise = b + noise;

%%
% Perform the DROP iterations.
tic;
options = struct();
options.lbound = 0;
options.ubound = 1;

Xdrop_noise = drop(A,b_noise,k,[],options);
toc
