function out = opt_score(Y,Yhat,Ystar)
%function [etaf, T] = opt_score(Tohat,eta,Nclasses)

% Tohat = Y (here)
% eta   = Yhat

[N,C] = size(Y);
D     = Y'*Y/N;
To    = diag(1./sqrt(diag(D)));
Tos   = Y*To; 

if nargin == 1   
    out = Tos; % output adjusted target values
else 
    Nstar = size(Ystar,2);
    
    Toshat = Yhat;   
    %Tohat = To; % assumes that the fit is perfect
    
    %[Phi,D] = eig(Tos'*Toshat);
    [Phi,S] = svd(Tos'*Toshat);
    eta  = Yhat*Phi(1:C,:)';
    etas = Ystar*Phi(1:C,:)';  % output adjusted predicted targets
    
    ic1 = Y(:,1) ~= 0;
    meta1 = [mean(eta(ic1,1)); 0];
    meta2 = [0; mean(eta(~ic1,2))];
         
    out = zeros(Nstar,1);
    for i = 1:size(etas,1)
        x1 = D*(etas(i,:)' - meta1);
        x2 = D*(etas(i,:)' - meta2);
        
        d1 = x1'*x1;
        d2 = x2'*x2;
        if d1 <= d2 
            out(i,1) = 1;
        end
        if d2 <= d1 
            out(i,2) = 1;
        end
    end
    out = out(:);
end

end