function K = sp_covMTL(cov, M, hyp, x, z, i)

% MTL covariance function 
% 
% usage:
%   sp_covMTL('init', number_of_hyperparameters);
%   sp_covMTL('initte', task_indicator_test);
%   sp_covMTL(input_covariance_function, task_indicator, ...); 
%
% takes a vector of hyperparameters:  
%   hyp = [ ell; covp ]
% where ell is the lower diagonal of L = chol(Kf)' and covp is a vector of
% parameters for the input covariance functions
%
%_________________________________________________________________________
% Copyright (C) A Marquand 

if nargin<2, error('Not enough parameters provided.'), end

persistent Nhyp;
persistent Mte;

if strcmp(cov,'init'), Nhyp = M; return; end
if strcmp(cov,'inittest'), Mte = M; return; end
if isempty(Nhyp), error('Covariance function not initialised.'); end

if nargin<4, K = num2str(Nhyp); return; end             % report number of parameters
if nargin<5, z = []; end                              % make sure, z exists
xeqz = numel(z)==0; 
dg   = ~iscell(z) && strcmp(z,'diag') && numel(z)>0;   % determine mode

T     = size(M,2);
lmaxi = (T*(T+1)/2);

% Reconstruct chol(Kf)' and Kf
Lf     = zeros(T);
lf     = hyp(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';

% configure input kernel from input arguments
if dg % kss
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
    if isempty(Mte)
        error('Covariance function not initialized for testing'); 
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