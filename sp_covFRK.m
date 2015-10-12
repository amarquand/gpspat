function K = sp_covFRK(param, hyp, x, z, i)

% MTL covariance function 
% 
% usage:
%   sp_covMTL('init', basis_covariance);
%   sp_covMTL('taskcov', basis functions, hyperparameters);
%   sp_covMTL(input_covariance_function, basis_functions, ...); 
%
% the spatial basis functions can be specified either as matrix S (N x R)
% (training) or as a cell array {S Ste} (testing)
% 
%
% this function requires a vector of hyperparameters:  
%   hyp = [ ell; covp ]
%
% where ell is the lower diagonal of L = chol(Kb)' and covp is a vector of
% parameters for the input covariance function (Kx)
%
%_________________________________________________________________________
% Copyright (C) A Marquand 

if nargin<1, error('Not enough parameters provided.'), end

persistent Nhyp;

if strcmp(param,'init'),
    if size(hyp,2) == 1
        Nhyp = size(hyp,1);
        error ('vector input is not implemented yet');
    else
        Nhyp = length( hyp(tril(ones(size(hyp,2))) ~= 0) );
    end
    return;
end
if isempty(Nhyp), error('Covariance function not initialised.'); end

if iscell(param)
    S = param{1};
else
    S = param;
end
R     = size(S,2);
lmaxi = (R*(R+1)/2);

if nargin<3,
    if strcmp(param,'taskcov')
        K = taskcov(hyp,lmaxi,R); % return task covariance matrix
    else 
        K = num2str(Nhyp);        % report number of parameters
    end
    return; 
end            
if nargin<4, z = []; end                              % make sure, z exists
xeqz = numel(z)==0; 
dg   = ~iscell(z) && strcmp(z,'diag') && numel(z)>0;   % determine mode

%Kb = taskcov(hyp,lmaxi,R);
[Kb,Lb,id] = taskcov(hyp,lmaxi,R);

% configure input kernel from input arguments
if dg % kss
  if iscell(param) && length(param) == 2
      Ste = param{2};
  else
      error('for diag mode, a cell array {S, Ste} is required.');
  end
  K  = Ste*Kb*Ste';
  K  = diag(K);
else
  if xeqz % K 
    %Kx = feval(cov{:}, hyp((lmaxi+1):end), x);
    %Kx = dokron(Kx,S);
    K  = (S*Kb*S');
  else % Ks
    if iscell(param) && length(param) == 2
      Ste = param{2};
    else
      error('for prediction, a cell array {S, Mtr} is required.');
    end
    %Kx = feval(cov{:}, hyp((lmaxi+1):end), x, z);
    K  = S*Kb*Ste';
  end
end

if nargin>=5  % return derivatives (otherwise returns covariances)
  if i <= Nhyp
     ID     = zeros(R);
     ID(id) = (1:lmaxi)';
     J      = double(ID == i);
   
     K = S*(J*Lb'+Lb*J')*S';
  else
    error('Unknown hyperparameter')
  end
end
end

function [Kb,Lb,id] = taskcov(hyp,lmaxi,R)

% Reconstruct chol(Kb)' and Kb
Lb     = zeros(R);
lb     = hyp(1:lmaxi);
id     = tril(true(R));
Lb(id) = lb;
Kb     = Lb*Lb';
end