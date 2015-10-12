function [varargout] = blr(hyp, X, t, xs)

% Bayesian linear regression
%
% Fits a bayesian linear regression model, where the inputs are:
%    hyp : vector of hyperparmaters. hyp = [log(beta); log(alpha)]
%    X   : N x D data matrix
%    t   : N x 1 vector of targets 
%    xs  : Nte x D matrix of test cases
%
% The hyperparameter beta is the noise precision and alpha is the precision
% over lengthscale parameters. This can be either a scalar variable (a
% common lengthscale for all input variables), or a vector of length D (a
% different lengthscale for each input variable, derived using an automatic
% relevance determination formulation).
%
% Two modes are supported: 
%    [nlZ, dnlZ, post] = blr(hyp, x, y);  % report evidence and derivatives
%    [mu, s2, post]    = blr(hyp, x, y, xs); % predictive mean and variance
%
% Written by A. Marquand

if nargin<3 || nargin>4
    disp('Usage: [nlZ dnlZ] = blr(hyp, x, y);')
    disp('   or: [mu  s2  ] = blr(hyp, x, y, xs);')
    return
end

[N,D]  = size(X);
beta   = exp(hyp(1));     % noise precision
alpha  = exp(hyp(2:end)); % weight precisions
Nalpha = length(alpha);
if Nalpha ~= 1 && Nalpha ~= D
    error('hyperparameter vector has invalid length');
end

if Nalpha == D
    %Sigma  = diag(alpha);      % weight prior precision
    %iSigma = diag(1./alpha);   % weight prior covariance
    Sigma  = diag(1./alpha);   % weight prior covariance
    iSigma = diag(alpha);      % weight prior precision
else    
    Sigma  = 1./alpha*eye(D);  % weight prior covariance
    iSigma = alpha*eye(D);     % weight prior precision
end

XX = X'*X;
A  = beta*XX + iSigma;     % posterior precision
Q  = A\X';
m  = beta*Q*t;             % posterior mean

if nargin == 3
    nlZ = -0.5*( N*log(beta) - N*log(2*pi) - log(det(Sigma)) ...
                 - beta*(t-X*m)'*(t-X*m) - m'*iSigma*m - log(det(A)) );
    
    if nargout > 1    % derivatives?
        dnlZ = zeros(size(hyp));
        b    = (eye(D) - beta*Q*X)*Q*t;
        
        % noise precision
        dnlZ(1) = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XX*m ...
                     - beta*b'*XX*m - b'*iSigma*m -0.5*trace(Q*X) )*beta;
        
        % variance parameters
        for i = 1:Nalpha
            if Nalpha == D % use ARD?
                dSigma      = zeros(D); 
                %dSigma(i,i) = 1              % if alpha is the variance
                dSigma(i,i) = -alpha(i)^-2;   % if alpha is the precision
            else
                dSigma = -alpha(i)^-2*eye(D);
            end
            
            F = -iSigma*dSigma*iSigma;
            c = -beta*F*X'*t;
            
            dnlZ(i+1) = -( -0.5*trace(iSigma*dSigma) + beta*t'*X*c - beta*c'*XX*m ...
                - c'*iSigma*m - 0.5*m'*F*m - 0.5*trace(A\F) )*alpha(i);
        end
        post.m = m;
        post.A = A;
    end
    if nargout > 1
        varargout = {nlZ, dnlZ, post};
    else
        varargout = {nlZ};
    end
    
else % prediction mode
    ys     = xs*m;
    s2     = 1/beta + diag(xs*(A\xs'));
    post.m = m;
    post.A = A;
    varargout = {ys, s2, post};
end

end