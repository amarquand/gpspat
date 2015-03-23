function K = covLINmtl(hyp, A, B, i)

% Linear covariance function with a single hyperparameter. The covariance
% function is parameterized as:
%
% k(x,z) = (x'*z + 1)/t2;
%
% The second term plays the role of the bias. The hyperparameter controls
% the scaling of the latent function and the bias:
%
% hyp = [ log(sqrt(t2)) ]
%__________________________________________________________________________
% This script is based on a script derived from the GPML toolbox, which is
% Copyright (C) Carl Rasmussen and Hannes Nickisch, 2011.

% Written by A Marquand 
% $Id: covLINkernel.m 176 2011-10-20 08:44:21Z amarquan $

%global Nhyp;
Nhyp = 2;%189;

if nargin<2, K = num2str(Nhyp); return; end             % report number of parameters
if nargin<3, B = []; end                              % make sure, B exists
xeqz = numel(B)==0; 
dg   = ~iscell(B) && strcmp(B,'diag') && numel(B)>0;   % determine mode

if iscell(A),
    Kx = A{1};
    M  = A{2};
else 
    error('covMTR: must supply a cell array with two arguments');
end
%if iscell(B), B = B{:}; end

%N     = size(Kx,1);
T     = size(M,2);
lmaxi = (T*(T+1)/2);

% Reconstruct chol(Kx)' and Kf
Lf     = zeros(T);
lf     = hyp(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';

% configure raw kernel from input arguments
if dg % kss
  K  = (M*Kf*M').*Kx;
  K = diag(K);
else
  if xeqz % K 
    %Kx = A;
    K = (M*Kf*M').*Kx;
  else % Ks
    Kx  = B{1};
    Mte = B{2};
    M   = B{3};
    K   = (Mte*Kf*M').*Kx;
    K   = K'; % transpose because gpml expects K(train,test)
  end
end

if nargin>=4  % return derivatives (otherwise returns covariances)
  if i <= Nhyp
     ID     = zeros(T);
     ID(id) = (1:lmaxi)';
     J      = double(ID == i);
   
     K = M*(J*Lf'+Lf*J')*M'.*Kx;
  else
    error('Unknown hyperparameter')
  end
end
