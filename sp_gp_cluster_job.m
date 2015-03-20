function [NLML, DNLML, Hyp, Yhat, S2, Yhattr, S2tr] = sp_gp_cluster_job(hyp0,X,Y,opt,Xs)

ones(10)*ones(10); % stupid hack to get matlab to work properly

%# function covLIN
%# function covLINard
%# function covSEiso
%# function covSEard
%# function covMaternard
%# function covMaterniso
%# function covSum
%# function meanPoly
%# function meanLinear
%# function meanConst
%# function covGrid
%# function infGrid
%# function sp_infGrid

T   = size(Y,2);  % number of tasks

% -----------------------------
% defaults
% -----------------------------
try opt.type2ml; catch, opt.type2ml = true; end
try opt.maxEval; catch, opt.maxEval = 100;  end
try opt.debug;   catch, opt.debug   = false;end

% % this is a hack. It it is included here because of the anonymous function
% if any(regexp(func2str(opt.inf),'infGrid'))
%     gopt.cg_maxit = 1000; gopt.cg_tol = 1e-3;
%     opt.inf       = @(varargin) infGrid(varargin{:},gopt);
% end

Hyp   = zeros(T,length(unwrap(hyp0)));
NLML  = zeros(T,1); 
DNLML = zeros(length(unwrap(hyp0)),T);

if nargin > 4 && nargout > 2
    N      = size(X,1);
    Ns     = size(Xs,1);
    Yhat   = zeros(Ns,T);
    S2     = zeros(Ns,T);
    Yhattr = zeros(N,T);
    S2tr   = zeros(N,T);
end
for t = 1:T
    if opt.debug; fprintf('processing case %d of %d ...\n',t,T); end
    y    = Y(:,t);
    hyp  = hyp0;
    nlml = NaN;
    
    if opt.type2ml
        try
            [hyp,nlml] = minimize(hyp, @gp, opt.maxEval, opt.inf, opt.mean, opt.cov, opt.lik, X, y);
        catch
            warning('Optimisation failed. Using default values');   
        end
    end
    if nargin > 4
        [yhat, s2] = gp(hyp,opt.inf,opt.mean,opt.cov,opt.lik, X, y, Xs, zeros(Ns,1));
        
        Yhat(:,t) = yhat;
        S2(:,t)   = s2;
        if nargout > 5
            [yhattr, s2tr] = gp(hyp,opt.inf,opt.mean,cov,opt.lik, X, y, X, zeros(N,1));
            Yhattr(:,t) = yhattr;
            S2tr(:,t)   = s2tr;
        end
    else % just report marginal likelihood and derivatives
        [nlml, dnlml] = gp(hyp,opt.inf,opt.mean,opt.cov,opt.lik, X, y);
        DNLML(:,t)    = unwrap(dnlml);
    end
    
    NLML(t)  = min(nlml);
    Hyp(t,:) = unwrap(hyp)';
end
