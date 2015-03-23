function out = opt_score(Y,Yhat,Ystar)
%function [etaf, T] = opt_score(Tohat,eta,Nclasses)

% Tohat = Y (here)
% eta   = Yhat

[N,C] = size(Y);
Nstar = size(Ystar);
D     = Y'*Y/N;
To    = diag(1./sqrt(diag(D)));
Tos   = Y*To; 

if nargin == 1   
    out = Tos; % output adjusted target values
else 
    Toshat = Yhat;   
    %Tohat = To; % assumes that the fit is perfect
    
    %[Phi,D] = eig(Tos'*Toshat);
    [Phi,S] = svd(Tos'*Toshat);
    eta  = Yhat*Phi(1:C,:)';
    etas = Ystar*Phi(1:C,:)';  % output adjusted predicted targets
    
    ic1 = Y(:,1) ~= 0;
    mc1 = mean(eta(ic1,1));
    mc2 = mean(eta(~ic1,2));
    
    eta = [repmat(mc1,Nstar,1
    
    dist = eta
    for i = 1:size(etas,1)
        
    end
end

end