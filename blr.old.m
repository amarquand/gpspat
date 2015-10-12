function [varargout] = blr(hyp, X, t, xs)

% Bayesian linear regression
%
% Fits a bayesian linear regression model, where the inputs are:
%    X is an N x D data matrix
%    t is an N x 1 vector of targets 
%    xs is an Nte x D matrix of test cases

if nargin<3 || nargin>4
    disp('Usage: [nlZ dnlZ] = blr(hyp, x, y);')
    disp('   or: [mu  s2  ] = blr(hyp, x, y, xs);')
        return
end

[N,D]  = size(X);
beta   = exp(hyp(1));     % noise precision
alpha  = exp(hyp(2:end)); % weight precisions
Sigma  = diag(alpha);     % weight prior covariance
iSigma = diag(1./alpha);  % weight prior precision

if size(X,2) ~= D
    
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
        for i = 1:D
            dSigma = zeros(D); dSigma(i,i) = 1;
            
            F = -iSigma*dSigma*iSigma;
            c = -beta*F*X'*t;
            
            dnlZ(i+1) = -( -0.5*trace(iSigma*dSigma) + beta*t'*X*c - beta*c'*XX*m ...
                - c'*iSigma*m - 0.5*m'*F*m - 0.5*trace(A\F) )*alpha(i);
        end
        post.m = m;
        post.A = A;
    end
    varargout = {nlZ, dnlZ, post};
    
else % prediction mode
    ys     = xs*m;
    s2     = 1/beta + diag(xs*(A\xs'));
    post.m = m;
    post.A = A;
    varargout = {ys, s2, post};
end

end