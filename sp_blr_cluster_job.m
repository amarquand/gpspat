function [NLML, DNLML, Hyp, Yhat, S2, Yhattr, S2tr] = sp_blr_cluster_job(hyp0,X,Y,opt,Xs)

ones(10)*ones(10); % stupid hack to get matlab to work properly

T   = size(Y,2);  % number of tasks

% -----------------------------
% defaults
% -----------------------------
try opt.type2ml; catch, opt.type2ml = true; end
try opt.maxEval; catch, opt.maxEval = 100;  end
try opt.debug;   catch, opt.debug   = false;end  
    
D = size(X,2);

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
    hyp  = zeros(D+1,1);
    nlml = NaN;
    
    if opt.type2ml
        try
            [hyp,nlml] = minimize(zeros(D+1,1), @blr, opt.maxEval, X, y);
            
            % % check gradients
            % fun   = @(lh)blr(lh,X,y);
            % [~,g] = blr(zeros(D+1,1),X,y);
            % gnum  = computeNumericalGradient(fun,zeros(D+1,1));
        catch
            warning('Optimisation failed. Using default values');   
        end
    end
    if nargin > 4
        [yhat, s2] = blr(hyp, X, y, Xs);
        
        Yhat(:,t) = yhat;
        S2(:,t)   = s2;
        if nargout > 5
            [yhattr, s2tr] = blr(hyp, X, y, X);
            Yhattr(:,t) = yhattr;
            S2tr(:,t)   = s2tr;
        end
    else % just report marginal likelihood and derivatives
         [nlml,DNLML(:,t)] = blr(hyp, X, y);
    end
    
    NLML(t)  = min(nlml);
    Hyp(t,:) = hyp';
end
