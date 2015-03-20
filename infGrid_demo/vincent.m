close all;
clear variables;
clc;

prompt = 'Which pattern would you like to extrapolate?\n1: Treadplate\n2: Wood\n3: Treadplate with NIPS Lettering and Lighting Non-Stationarities\n';
result = input(prompt);


% Load Data

if(result==1)
  A = imread(['treadplate'], 'jpg');
  I = rgb2gray(A);
  II  = imresize(I, 1/12);
  imagedata  = double(II(1:100,1:100));
elseif(result==2)
  A = imread(['wood'], 'png');
  I = rgb2gray(A);
  II  = imresize(I, 1/12);
  imagedata  = double(II(1:100,1:100));
elseif(result==3)
  A = imread(['treadplate'], 'jpg');
  I = rgb2gray(A);
  II  = imresize(I, 1/12);
  imagedata  = double(II(1:100,1:100));

  A = imread(['NIPS'], 'png');  
  I = rgb2gray(A);

  [X1,X2] = meshgrid(1:size(imagedata,1),1:size(imagedata,2));
  imagedata = imagedata.*sqrt(X1.^2+X2.^2);
else
  error('Invalid Input.  Enter 1, 2, or 3.');
end

fprintf('Thank you.  This program will complete in 5-10 minutes.\n\n')

% --- make missing region --- 

if(result <=2)
  mask = ones(size(imagedata));
  mask(25:75,25:75) = 0;
elseif(result==3)
  mask=I<200;
  mask(size(imagedata,1),size(imagedata,2))=0;
  mask = ~mask;
end
% -----------------




% define each grid dimension
xg = {(1:size(imagedata,1))',(1:size(imagedata,2))'};

% conventional N x P input vector, for N datapoints and P total
% input dimensions
x = covGrid('expand',xg);
N = size(x,1);

% Show the image.  We wish to denoise the training (visible) region, and 
% extrapolate the the missing (black) region.
figure(1);
imagesc(imagedata.*mask);colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')
title('Training Data (Black = Missing)')

% Label the black in the image as missing data.
% These missing regions will be the locations of the virtual observations.
mask = double(mask);
mask(mask==false)=NaN; 
tmp = isnan(mask);
tmp = tmp(:);
idx = find(~tmp);  % training input locations
idxstar = find(tmp); % missing input locations

% stack columns of imagedata
data = imagedata(:);  % pattern including missing regions.

% training data.  because of the missing regions, this data is not on a 
% grid, but we can still exploit the partial grid structure for efficiency.
y = data(idx); 

% verify that the data stores all of the image as a vector
% figure(2); imagesc(reshape(data,[100 100]))

% standardise data
my = mean(y);
sy = std(y);
y = ((y - my) ./ sy);

% Specify a spectral mixture kernel in each input dimension, each with 
% Q components.
Q = 20;
cov = {{'covSMfast',Q},{'covSMfast',Q}};  % 1D SM kernel for each input dimension, with Q components
covg = {@covGrid,cov,xg};

% mean function for the GP  
gpmean = {@meanConst}; 
hyp.mean=mean(y);  % if normalised, mean(y) will be zero.

% specify that we want to use inf grid
% for missing data we need to run LCG.  we specify here the maximum 
% number of iterations, and the tolerance.  
% a more conservative setting would be opt.cg_maxit = 600; opt.cg_tol=1e-4
opt.cg_maxit = 200; opt.cg_tol = 1e-2; 
if(result==2 || result==3)
    opt.cg_maxit = 400;  % use more iterations for the harder problem
                         % more missing data typically requires more LCG
                         % iterations.
end


inf_method = @(varargin) infGrid(varargin{:},opt);
lik = @likGauss;




% Initialise the noise standard deviation
% 10 percent of the mean(abs(y)) is typically a good initialisation
sn = .1*mean(abs(y));  
hyp.lik = log(sn);

% initialise spectral mixture kernel hyperparameters
inits = 100; % 10 initialisations are usually sufficient for good results.  But these 
             % are cheap to evaluate, so might as well try many.
hyp = spectral_init(inf_method,hyp,gpmean,lik,cov,covg,x,y,idx,inits);


iters = 700;

% If you wish to use BFGS instead of non-linear conjugate gradients.
% BFGS typically finds better solutions more quickly, but will give up more 
% easily than the non-linear conjugate gradients implementation.
%{
% change to BGFS
p.length =    -iters;
p.method =    'BFGS';  % 'BFGS' 'LBFGS' or 'CG'
p.SIG = 0.1;
p.verbosity = 2; %0 quiet, 1 line, 2 line + warnings (default), 3 graphical
%             p.mem        % number of directions used in LBFGS (default 100)
          
tic;  
%hyp1 = minimize_new(hyp,@gp,p,inf_method,gpmean,covg,lik,idx,y);
toc;

%}

% minimize using non-linear conjugate gradients.

% final nlml for treadplate should be near 3.4e3.  If trained nlml > 4e3 
% then restart this script. Training time on a home PC should be no more than 600s.
% final nlml for treadplate with letters should be near -7.85e3.
tic
hyp1 = minimize(hyp,@gp,-iters,inf_method,gpmean,covg,lik,idx,y);
toc


tic
[postg nlZg dnlZg] = infGrid(hyp1, gpmean, covg, 'likGauss', idx, y, opt);
toc

% Here we do not need variance predictions, just predictive means.  
% So to speed things up we set L to an efficient function to evaluate.
% Comment this postg.L specification if you want the proper predictive variances.
% Note that one can only use this trick for symmetric likelihoods
postg.L = @(x) 0*x;   

% indices of points where we wish to make predictions
% make predictions at all N input locations (training and testing)
star_ind = (1:N)';

tic
[ymug ys2g fmug fs2g] = gp(hyp1, @infGrid, gpmean, covg, [], idx, postg, star_ind);
toc

% adjust for prior data normalisation
ymug = ymug*sy + my;

ypred = reshape(ymug,[100 100]);

figure(3);
imagesc(ypred);colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')
title('Restored Image with GPatt Extrapolation');

figure(4);
imagesc(reshape(data,[100 100]));
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')
title('Original Full Image');


% Let's try with the SE kernel

disp('Comparing to covSE extrapolation')

cov_SE = {{'covSEiso'},{'covSEiso'}};  % 1D SE kernel for each input dimension
covg_SE = {@covGrid,cov_SE,xg};

% initialise SE kernel hyperparameters

sn_SE = .1*mean(abs(y));  
hypSE.lik = log(sn_SE);
ell = 20;    % initial length-scale
sf = std(y); % initial signal standard deviation
hypSE.cov = log([ell; sf; ell; sf]);

SE_iters = 1000;

% Set mean function
gpmean_SE = {@meanConst}; 
hypSE.mean=mean(y);  % if normalised, mean(y) will be zero.

tic
hypSE1 = minimize(hypSE,@gp,-SE_iters,inf_method,gpmean_SE,covg_SE,lik,idx,y);
toc


[postg_SE nlZgSE dnlZgSE] = infGrid(hypSE1, gpmean_SE, covg_SE, 'likGauss', idx, y, opt);

% indices of points where we wish to make predictions
% make predictions at all N input locations (training and testing)
star_ind = (1:N)';

postg_SE.L = @(x) 0*x;   % Comment this line if you want predictive variances

[ymugSE ys2gSE fmugSE fs2gSE] = gp(hypSE1, @infGrid, gpmean_SE, covg_SE, [], idx, postg_SE, star_ind);

% adjust for prior data normalisation
ymugSE = ymugSE*sy + my;

ypredSE = reshape(ymugSE,[100 100]);


% All together now!

figure(5)
clf;
subplot(331), imagesc(imagedata.*mask); title('Training Data')
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')


reverseMask = mask;
reverseMask(reverseMask==1) = 0;
reverseMask(isnan(reverseMask)) = 1;
reverseMask(reverseMask==0) = NaN;

figure(5);
subplot(332), imagesc(imagedata.*reverseMask); title('Withheld Data');
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')

subplot(333), imagesc(reshape(data,[100 100])); title('Full Pattern')
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')


subplot(334), imagesc(ypred); title('GPatt (covSM) Extrapolation')
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')



subplot(335), imagesc(reshape(ymugSE,[100 100])); title('covSE extrapolation')
colormap(gray);
set(gca,'Xtick',[],'XTickLabel','')
set(gca,'Ytick',[],'YTickLabel','')

% Plot the learned kernels
SM1 = cov{1};
SM2 = cov{2};

krange = [0:1:99]';

k_SM1 = feval(SM1{:},hyp1.cov(1:3*Q),0,krange);
k_SM2 = feval(SM1{:},hyp1.cov(3*Q+1:end),0,krange);

figure(5); 
subplot(337); plot(k_SM1); xlabel('\tau'); ylabel('Covariance'); title('Learned covSM Dim 1');
subplot(338); plot(k_SM2); xlabel('\tau'); ylabel('Covariance'); title('Learned covSM Dim 2');

SE1 = cov_SE{1};
SE2 = cov_SE{2};

k_SE1 = feval(SE1{:},hypSE1.cov(1:2),0,krange);
k_SE2 = feval(SE2{:},hypSE1.cov(3:4),0,krange);

figure(5);
subplot(336); plot(k_SE1); xlabel('\tau'); ylabel('Covariance'); title('Learned covSE Dim 1');
subplot(339); plot(k_SE2); xlabel('\tau'); ylabel('Covariance'); title('Learned covSE Dim 2');







