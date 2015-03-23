function [Acc] = gp_da_run(c1,c2,cov,output_name,opt)

% %clsnum = [1,2];
% c1  = 1; c2 =2; 
% cov = 'lin';
% output_name = ['../gp_da/t2ml_',mat2str(c1),'_v_',num2str(mat2str(c2)),'_YY0'];
% opt.maxEval = 500;
% opt.optimiseTheta = false;

addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');

% defaults
try opt.optimiseTheta;  catch, opt.optimiseTheta = true;   end
try opt.computeWeights; catch, opt.computeWeights = false; end
try opt.normalizeK;     catch, opt.normalizeK = false;     end
try opt.maxEval;        catch, opt.maxEval = 500;        end

[Xa,Ya,ID,~,classes] = load_data;

% get rid of fMRI tasks we are not considering
Ya = [sum(Ya(:,c1),2), sum(Ya(:,c2),2)]; %Ya = Ya(:,clsnum);
id = sum(Ya,2) > 0;
Ya = Ya(id,:);
Xa = Xa(id,:);
ID = ID(id,:);

[X,y,ID,Y,Ys] = process_tasks(Xa,Ya,ID);
[N, T]        = size(Y);
Nclassifiers  = max(ID(:,1));
Nfolds        = length(unique(ID(:,2)));

[~, y_kfid] = find(Ys);
y_kxid      = 1:N;

% starting hyperparamers
Kf00  = eye(T);
kf00 = Kf00(tril(ones(T)) ~= 0);

% options
switch cov
    case 'se'
        error('Squared exponential not adjusted yet');
        opt.CovFunc = 'covfunc_mtr_se'; kx0 = [log(1); log(1000); log(0.1*ones(T,1))];
        kx0 = [log(1); log(1); log(0.1*ones(T,1))];
        lh00 = [kf00; kx0];
    otherwise % linear
        opt.CovFunc = 'covfunc_mtr_nonblock'; kx0 = log(0.1*ones(T,1));
        %opt.CovFunc = 'covfunc_mtr'; kx0 = log(ones(T,1));
        lh00 = [kf00; kx0];
end

% ----------------Main cross-validation loop----------------------------
trall  = zeros(N,Nfolds);
teall  = zeros(N,Nfolds);
yhattr = cell(Nfolds,1);
yhatte = cell(Nfolds,1);
s2te   = cell(Nfolds,1);
Alpha  = zeros(Nfolds,T,N);
Hyp    = zeros(Nfolds,length(lh00));
%matlabpool('open');
%par
for f = 1:Nfolds
    fprintf('Outer loop %d of %d ...\n',f,Nfolds)
    optf = opt;
    
    sid = ID(:,2) == f;
    if sum(sid) == 0, error(['No tasks found for fmri run ', num2str(r)]); end
    te = find(sid);
    tr = find(~sid);
    teall(:,f) = sid;
    trall(:,f) = ~sid;
       
    % training mean
    Ystr = Ys(tr,:);
    %Ytr = Y(tr,:);
    %mtr = mean(Y(tr,:)); Mtr = repmat(mtr,length(tr),1); Ytr = Ytr - Mtr;
    
    %fprintf('Standardising features ...\n ')
    Xz = (X - repmat(mean(X(tr,:)),N,1)) ./ repmat(std(X(tr,:)),N,1);
    Xz = Xz(:,logical(sum(isfinite(Xz))));
    Phi = Xz*Xz';
    if strcmp(opt.CovFunc,'covfunc_mtr_se') || opt.normalizeK
        disp('Normalizing kernel ...');
        Phi = prt_normalise_kernel(Phi);
    end
        
    if opt.optimiseTheta
        % set initial hyperparameter values
        %Kf0 = eye(T);
        Kf0 = pinv(Ystr)*(y(tr)*y(tr)' ./ Phi(tr,tr))*pinv(Ystr)'; optf.Psi_prior = Kf0;
        kf0 = Kf0(tril(ones(T)) ~= 0);
        lh0 = [kf0; kx0];
        
        [hyp nlmls] = minimize(lh0, @gp_mtr, opt.maxEval, {Phi(tr,tr), Ystr}, {y(tr)}, opt);
        %[hyp nlmls] = minimize(lh0, @gp_mtr_map, opt.maxEval, {Phi(tr,tr), Ytr}, Ytr, optf);
    else
        Kf0 = eye(T);
        kf0 = Kf0(tril(ones(T)) ~= 0);
        hyp = [kf0; kx0];
    end
    
    C   = feval(opt.CovFunc,{Phi, Ys},hyp);
    K   = C(tr,tr);
    Ks  = C(te,tr);
    kss = C(te,te);
    
    [yhatte{f}, s2f] = gp_pred_mtr(K,Ks,kss,{y(tr)});
    yhattr{f}        = gp_pred_mtr(K,K,K,{y(tr)});
   
    %yhatte{te}  = yhattte{f}';% + mtr;
    s2te{f}    = diag(s2f);
    Hyp(f,:)   = hyp';
    
    % weights
    if opt.computeWeights
        disp('Computing weights ...');
        mask        = '/cns_zfs/mlearn/public_datasets/openfmri/posner/masks/SPM_mask_46x55x39.img';
        [~,~,alpha] = gp_mtr(hyp, {Phi(tr,tr), Ystr}, {y(tr)}, opt);
        nvox        = size(X,2);
        
        Wm  = alpha'*Xz(tr,:);
        Wmn = Wm ./ norm(Wm); % for visualisation only 
        prt_write_nii(Wmn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_meantask.img']);
        
        W  = zeros(T,nvox);
        for t = 1:T
            xid = find(ID(:,1) == t & ID(:,2) ~= f);
            aid = find(ID(tr,1) == t);
            
            W(t,:) = alpha(aid)'*Xz(xid,:);
            
            wn = W(t,:) ./ norm(W(t,:));
            prt_write_nii(wn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(t),'.img']);
        end
    end

    fprintf('Outer loop %d of %d done.\n',f,Nfolds)
end
matlabpool('close')

% reconstruct predictions
yhat   = zeros(N,1); 
s2     = zeros(N,1);
for f = 1:Nfolds
   te = find(teall(:,f));
   yhat(te) = yhatte{f};
   s2(te)   = s2te{f};
end

% Reconstruct chol(Kx)' and Kf
lmaxi = (T*(T+1)/2);
Noise = zeros(Nfolds,T); 
Kf    = zeros(T,T,Nfolds);
for f = 1:Nfolds
    Lf        = zeros(T);
    lf        = Hyp(f,1:lmaxi)';
    id        = tril(true(T));
    Lf(id)    = lf;
    Kf(:,:,f) = Lf*Lf';
    Noise(f,:) = exp(Hyp(f,end-T+1:end));
end

% compute accuracy
Acc    = zeros(Nclassifiers,1);
Acc05  = zeros(Nclassifiers,1);
for c = 1:Nclassifiers
    %c
    clsid = find(ID(:,4) == c);
    trlab = y(clsid) ~= 0;
    
    Yf = [y(clsid) y(clsid(end:-1:1))];
    
    prlab    = opt_score(Yf,Yf,yhat(clsid));
    prlab05  = yhat(clsid) > 0.5;
    
    Acc(c)   = sum(trlab == prlab) ./ length(trlab);
    Acc05(c) = sum(trlab == prlab05) ./ length(trlab);
end
fprintf('Mean accuracy (OS): %02.2f\n',mean(Acc))
fprintf('Mean accuracy (LR): %02.2f\n',mean(Acc05))

save(output_name ,'y','yhat','Acc','Hyp','Noise','Kf','Alpha','ID')

% check gradients
% % fun   = @(lh)gp_mtr(lh,X,Y,opt);
% % [~,g] = gp_mtr(lh0,X,Y,opt);
% fun   = @(lh)gp_mtr(lh,{Phi, Y},Y,opt);
% [~,g] =gp_mtr(lh0,{Phi, Y},Y,opt);
% gnum  = computeNumericalGradient(fun,lh0);