function K = sp_covMTL(cov, param, hyp, x, z, i)

% MTL covariance function 
% 
% usage:
%   sp_covMTL('init', number_of_hyperparameters);
%   sp_covMTL('taskcov', task_indicator, hyperparameters);
%   sp_covMTL(input_covariance_function, task_indicator(s), ...); 
%
% the task indicators can be specified either as matrix M (training) or as
% a cell array {Mtr, Mte} (testing)
%
% this function requires a vector of hyperparameters:  
%   hyp = [ ell; covp ]
%
% where ell is the lower diagonal of L = chol(Kf)' and covp is a vector of
% parameters for the input covariance function (Kx)
%
%_________________________________________________________________________
% Copyright (C) A Marquand 

if nargin<2, error('Not enough parameters provided.'), end

persistent Nhyp;

if strcmp(cov,'init'), Nhyp = param; return; end
if isempty(Nhyp), error('Covariance function not initialised.'); end

if iscell(param)
    M = param{1};
else
    M = param;
end
T     = size(M,2);
lmaxi = (T*(T+1)/2);

if nargin<4,
    if strcmp(cov,'taskcov')
        K = taskcov(hyp,lmaxi,T); % return task covariance matrix
    else 
        K = num2str(Nhyp);        % report number of parameters
    end
    return; 
end            
if nargin<5, z = []; end                              % make sure, z exists
xeqz = numel(z)==0; 
dg   = ~iscell(z) && strcmp(z,'diag') && numel(z)>0;   % determine mode

[Kf,Lf,id] = taskcov(hyp,lmaxi,T);

% configure input kernel from input arguments
if dg % kss
  if iscell(param) && length(param) == 2
      Mte = param{2};
  else
      error('for diag mode, a cell array {M, Mtr} is required.');
  end
  Kx = feval(cov{:}, hyp((lmaxi+1):end), x);
  %K  = (M*Kf*M').*Kx;
  K  = (Mte*Kf*Mte').*Kx;
  K  = diag(K);
else
  if xeqz % K 
    Kx = feval(cov{:}, hyp((lmaxi+1):end), x);
    Kx = dokron(Kx,M);
    K  = (M*Kf*M').*Kx;
  else % Ks
    if iscell(param) && length(param) == 2
      Mte = param{2};
    else
      error('for prediction, a cell array {M, Mtr} is required.');
    end
    Kx = feval(cov{:}, hyp((lmaxi+1):end), x, z);
    %K  = (Mte*Kf*M').*Kx;
    %K  = K'; % transpose because gpml expects K(train,test)
    K  = (M*Kf*Mte').*Kx;
  end
end

if nargin>=6  % return derivatives (otherwise returns covariances)
  if i <= Nhyp
     ID     = zeros(T);
     ID(id) = (1:lmaxi)';
     J      = double(ID == i);
   
     K = M*(J*Lf'+Lf*J')*M'.*Kx;
  else
    error('Unknown hyperparameter')
  end
end
end

function Kx = dokron(Kx,M)

[Nm,T] = size(M);
Nx     = size(Kx,1);

if Nm == Nx 
    return;
elseif Nx*T == Nm
    Kx = repmat(Kx,T,T);
else
    error('Input kernel and indicator matrix have incompatible sizes');
end
end

function [Kf,Lf,id] = taskcov(hyp,lmaxi,T)

% Reconstruct chol(Kf)' and Kf
Lf     = zeros(T);
lf     = hyp(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';
end