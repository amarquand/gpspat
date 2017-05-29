function [out1, out2, post] = blr_multi(hyp, X, T, xs)

% Bayesian linear regression (multiple independent targets)
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
% The main difference between this version and the vanilla version of blr
% is that this version precomputes lots of quantities that are used
% repeatedly for computing i.i.d samples with the same posterior covariance
% (i.e. when T is a matrix). for such cases this is more efficient than
% computing each separately.
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
Nrep   = size(T,2);
beta   = exp(hyp(1));     % noise precision
alpha  = exp(hyp(2:end)); % weight precisions
Nalpha = length(alpha);
if Nalpha ~= 1 && Nalpha ~= D
    error('hyperparameter vector has invalid length');
end

if Nalpha == D
    Sigma    = diag(1./alpha);   % weight prior covariance
    invSigma = diag(alpha);      % weight prior precision
else    
    Sigma    = 1./alpha*eye(D);  % weight prior covariance
    invSigma = alpha*eye(D);     % weight prior precision
end
Sigma = sparse(Sigma);
invSigma = sparse(invSigma);

% invariant quantities that do not need to be recomputed each time
XX   = X'*X;
A    = beta*XX + invSigma;     % posterior precision
S    = inv(A);                 % posterior covariance. Store for speed
Q    = S*X';
%Q    = A\X';
trQX = trace(Q*X);
R    = (eye(D) - beta*Q*X)*Q;

% compute like this to avoid numerical overflow
logdetA     = 2*sum(log(diag(chol(A))));
logdetSigma = sum(log(diag(A)));            % assumes Sigma is diagonal

% save posterior precision
post.A = A;

for r = 1:Nrep
    %if mod(r,5) == 0, fprintf('%d ',r); end
    t = T(:,r);               % targets
    m = beta*Q*t;             % posterior mean
    % save posterior means
    if r == 1, post.M = zeros(length(m), Nrep); end
    post.M(:,r) = m;
    
    % frequently needed quantities dependent on t and m
    Xt  = X'*t;
    XXm = XX*m;
    SXt = S*Xt;
    
    if nargin == 3  
        if r == 1, NLZ = zeros(Nrep,1);  end
        
        NLZ(r) = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
                 - beta*(t-X*m)'*(t-X*m) - m'*invSigma*m - logdetA );

        if nargout > 1    % derivatives?
            if r == 1
                DNLZ = zeros(length(hyp), Nrep);
            end
            b  = R*t;
            
            % noise precision
            DNLZ(1,r) = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m  ...
                        + beta*t'*X*b - 0.5*m'*XX*m - beta*b'*XX*m ...
                        - b'*invSigma*m -0.5*trQX )*beta;
            
            % variance parameters
            for i = 1:Nalpha
                if Nalpha == D % use ARD?
                    dSigma = sparse(i,i,-alpha(i)^-2,D,D);
                    dinvSigma = sparse(i,i,1,D,D);
                else
                    dSigma = -alpha(i)^-2*eye(D);
                    dinvSigma = eye(D);
                end
                
                %F = -invSigma*dSigma*invSigma;
                %c = -beta*F*Xt;
                F = dinvSigma;
                c = -beta*S*F*SXt;
                
                DNLZ(i+1,r) = -(-0.5*sum(sum(invSigma.*dSigma')) + ...
                                beta*Xt'*c - beta*c'*XXm - c'*invSigma*m ...
                                - 0.5*m'*F*m - 0.5*sum(sum(S*F')) ...
                                )*alpha(i);
            end
        end       
    else % prediction mode
        if r == 1
            Ys = zeros(size(xs,1),Nrep);
            S2 = zeros(size(xs,1),Nrep);
            s2 = 1/beta + sum((xs*S).*xs,2); % assumes that xs is constant
        end
        Ys(:,r) = xs*m;
        S2(:,r) = s2;
        %S2(:,r) = 1/beta + diag(xs*(A\xs')); % sloooow
    end
end
%fprintf('\n');

% use this syntax instead of varargout to be able to compile this function
if nargin == 3
    out1 = sum(NLZ);
    if nargout > 1
        out2 = sum(DNLZ,2);
    else
        out2 = [];
    end
else
    out1 = Ys;
    out2 = S2;
end
end