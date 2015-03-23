function out = opt_score(Y,Yhat,Ystar)
%function [etaf, T] = opt_score(Tohat,eta,Nclasses)

% Tohat = Y (here)
% eta   = Yhat

[N,C] = size(Y);

if C ~= 2
    error('only implemented for binary classification')
end
D     = Y'*Y/N;
To    = [1./sqrt(diag(D(1,1))); 0];
Tos   = Y*To; 

if nargin == 1   
    out = Tos; % output adjusted target values
else
    Nstar = size(Ystar,1); 
    
    Toshat = Yhat;   
    %Tohat = To; % assumes that the fit is perfect
    
    %[Phi,D] = eig(Tos'*Toshat);
    [Phi,S] = svd(Tos'*Toshat);
    
    eta  = Yhat*Phi';
    etas = Ystar*Phi';  % output adjusted predicted targets
    
    ic1 = Tos ~= 0;
    meta1 = mean(eta(ic1,1));
    meta2 = 0;
    
    out = zeros(Nstar,1);    
    for i = 1:size(etas,1)
        x1 = D*(etas(i,:)' - meta1);
        x2 = D*(etas(i,:)' - meta2);
        
        d1 = x1'*x1;
        d2 = x2'*x2;
        if d1 <= d2
            out(i,1) = 1;
        end
    end
    out = out(:);
    
end

end