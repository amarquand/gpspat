%function [] = gp_da_run(clsnum,cov,output_name,opt)

clsnum = [2,4];
cov = 'lin';
output_name = '../gp_da/test1';
opt.maxEval = 500;

addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');

[Xa,Ya,ID,mask,classes] = load_data;

% defaults
try opt.computeWeights; catch, opt.computeWeights = false; end
try opt.normalizeK;     catch, opt.normalizeK = false;     end
try opt.maxEval;        catch, opt.maxEval = -5000;        end
try clsnum;             catch, clsnum = 1:size(Y,2);       end

% get rid of fMRI tasks we are not considering
Ya = Ya(:,clsnum);
id = sum(Ya,2) > 0;
Ya = Ya(id,:);
Xa = Xa(id,:);
ID = ID(id,:);

% define some variables to improve readability
subjects  = unique(ID(:,1));
fmriRuns  = unique(ID(:,2));
fmriTasks = unique(ID(:,3));

disp('Checking for missing scans ...');
bad = false(size(ID,1),1);
for s = 1:length(subjects)
    sid = ID(:,1) == s;
    for r = 1:length(unique(ID(sid,2)))
        srid = sid & ID(:,2) == r;
        
        if numel(unique(ID(srid,3))) < size(Ya,2)
            fprintf('> Excluding subject %d, run %d (missing data)\n',subjects(s),r)
            bad(srid) = true;
        end
    end
end
disp('Error checking done.');
ID = ID(~bad,:);
Ya = Ya(~bad,:);
Xa = Xa(~bad,:);

% Ya contains one column per fmri task. Now we reparametrise so that Y is
% a cell array containing one cell per subject per fmri task. We also
% duplicate the data matrix to accommodate the binary classification.
%
% NOTE: at this stage the noise parameters for each class within each
%       subject are NOT coupled!
%
T   = size(Ya,2)*length(subjects);
N   = size(Ya,2)*length(ID(:,1));
Yc  = cell(1,T);
Y   = zeros(N,T);
X   = zeros(N,size(Xa,2));
IDm = kron(ID,ones(size(Ya,2),1));
ct  = 1; 
for s = 1:length(subjects)
    for t = 1:length(fmriTasks)
         
        sid = ID(:,1) == subjects(s);
        Yc{ct} = Ya(sid,t)';
        Yc{ct} = Ya(sid,t)';
        
        sidmtl = IDm(:,1) == subjects(s) & IDm(:,3) == fmriTasks(t);
        Y(sidmtl,ct) = 1;
        X(sidmtl,:)   = Xa(sid,:);
        IDm(sidmtl,4) = ct;
        ct = ct + 1;
    end
end

% Optimal scoring
Nclassifiers = max(IDm(:,1));
cls = 1;
for c = 1:Nclassifiers
    Yos       = opt_score_2latents([Yc{cls}', Yc{cls+1}']);
    Yc{cls}   = Yos(:,1)';
    Yc{cls+1} = Yos(:,2)';
    cls       = cls + 2;
end

y = [Yc{:}]'; % Y(:) ~= y !!!

% starting hyperparamers
Kf0 = eye(T);
Tm  = tril(ones(T));
kf0 = Kf0(Tm ~= 0);

% options
switch cov
    case 'se'
        error('Squared exponential not adjusted yet');
        opt.CovFunc = 'covfunc_mtr_se'; kx0 = [log(1); log(1000); log(0.1*ones(T,1))];
        kx0 = [log(1); log(1); log(0.1*ones(T,1))];
        lh0 = [kf0; kx0];
    otherwise % linear
        opt.CovFunc = 'covfunc_mtr_nonblock'; kx0 = log(0.1*ones(T,1));
        %opt.CovFunc = 'covfunc_mtr'; kx0 = log(ones(T,1));
        lh0 = [kf0; kx0];
end

% Configure Cross-validation parameters
Nfolds = length(fmriRuns);
trall  = zeros(N,Nfolds);
teall  = zeros(N,Nfolds);
ytrall = cell(1,N);
yteall = cell(1,N);
for r = 1:length(fmriRuns)
    %sid = ID(:,2) == fmriRuns(r);
    sid = IDm(:,2) == fmriRuns(r);
    
    teall(:,r) = sid;
    trall(:,r) = ~sid;
    yteall{r}  = y(sid);
    ytrall{r}  = y(~sid);
    
    if sum(sid) == 0, error(['No tasks found for fmri run ', num2str(r)]); end
end

% ----------------Main cross-validation loop----------------------------
yhat   = zeros(N,1); 
s2     = zeros(N,1);
Hyp    = zeros(Nfolds,length(lh0));
yhattr = cell(Nfolds,1);
Alpha  = zeros(Nfolds,T,N);
%matlabpool('open');
%par
for f = 1:Nfolds
    fprintf('Outer loop %d of %d ...\n',f,Nfolds)
    optf = opt;
    
    tr = find(trall(:,f));
    te = find(teall(:,f));
    
    %fprintf('Standardising features ...\n ')
    Xz = (X - repmat(mean(X(tr,:)),N,1)) ./ repmat(std(X(tr,:)),N,1);
    Phi = Xz*Xz';
    if strcmp(opt.CovFunc,'covfunc_mtr_se') || opt.normalizeK
        disp('Normalizing kernel ...');
        Phi = prt_normalise_kernel(Phi);
    end
    %Kf0 = 1/N * (Y - repmat(mean(Y),N,1))'/(Phi+1e-5*eye(size(Phi)))*(Y - repmat(mean(Y),N,1)); optf.Psi_prior = Kf0;
    kf0 = Kf0(Tm ~= 0);
    lh0 = [kf0; kx0];
        
    % training mean
    Ytr = Y(tr,:);
    %mtr = mean(Y(tr,:));
    %Mtr = repmat(mtr,length(tr),1);
    %Ytr = Ytr - Mtr;
    
    hyp = lh0;
    %[hyp nlmls] = minimize(lh0, @gp_mtr, opt.maxEval, {Phi(tr,tr), Ytr}, {ytrall{f}}, opt);
    %[hyp nlmls] = minimize(lh0, @gp_mtr_map, opt.maxEval, {Phi(tr,tr), Ytr}, Ytr, optf);
    
    C   = feval(opt.CovFunc,{Phi, Y},hyp);
    K   = C(tr,tr);
    Ks  = C(te,tr);
    kss = C(te,te);
    
    [yhat(te), s2f] = gp_pred_mtr(K,Ks,kss,{ytrall{f}});
    yhattr{f}       = gp_pred_mtr(K,K,K,{y(tr)});
   
    %yhat(te)  = yhatf';% + mtr;
    s2(te)    = diag(s2f);
    Hyp(f,:) = hyp';
    
    % weights
    if opt.computeWeights
        error('Weights not adjusted yet');
        disp('Computing weights ...');
        [~,~,alpha] = gp_mtr(hyp, {Phi(tr,tr), Ytr}, Ytr, opt);
        a = zeros(N,T); a(tr,:) = reshape(alpha,N-1,T);
        Alpha(:,:,f) = a;
        mask        = '/home/kkvi0203/svmdata/PD_MSA_PSP/spgrc/shoot_scalmom_spm-8-4667_69_subjects/smooth_10mm/gmwm_mask_final.img';
        nvox        = size(X,2) / 2;
        
        %%W = reshape(alpha,N-1,T)'*Xz(tr,:);
        %Xa = repmat(Xz(tr,:),T,T);
        %W  = Xa'*alpha;
        %W  = reshape(W,2*nvox,T)';
        %clear Xa;
        
        for w = 1:size(W,1)
            WW = [W(w,1:nvox); W(w,nvox+1:end)];
            XsW = [Xz(f,1:nvox); Xz(f,nvox+1:end)].* WW;
            
            WW = WW ./ norm(WW); % for visualisation only 
            prt_write_nii(WW,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(w),'.img']);
            %prt_write_nii(XsW,mask,[output_name,'_XW_fold_',num2str(f,'%02.0f'),'_task',num2str(w),'.img']);
        end
    end

    fprintf('Outer loop %d of %d done.\n',f,Nfolds)
end
%matlabpool('close')

% Reconstruct chol(Kx)' and Kf
lmaxi = (T*(T+1)/2);
Noise = zeros(Nfolds,T); Kf = zeros(T,T,Nfolds);
for f = 1:Nfolds
    Lf        = zeros(T);
    lf        = Hyp(f,1:lmaxi)';
    id        = tril(true(T));
    Lf(id)    = lf;
    Kf(:,:,f) = Lf*Lf';
    Noise(f,:) = exp(Hyp(f,end-T+1:end));
end

Nclass = 2;
cls    = 1;
Acc    = zeros(Nclassifiers,1);
Acc05  = zeros(Nclassifiers,1);
for c = 1:Nclassifiers
    c
    clsid = find(IDm(:,4) == cls | IDm(:,4) == cls+1);
    trlab = y(clsid) ~= 0;
    
    Yco      = reshape(y(clsid), length(clsid)/Nclass, Nclass);
    Yhatco   = reshape(yhat(clsid), length(clsid)/Nclass, Nclass);
    %Yhattrco = reshape(yhattr{c}(clsid), length(clsid)/Nclass, Nclass);
    %%P        = opt_score(Yco,Yhattrco,Yhatco);
    % P        = opt_score(Yco,Yco,Yhatco);
    
    prlab = opt_score_2latents(Yco,Yco,Yhatco);
    prlab05 = yhat(clsid) > 0.5;
    Acc(c)   = sum(trlab == prlab) ./ length(trlab);
    Acc05(c) = sum(trlab == prlab05) ./ length(trlab);
    
    cls = cls + 2;
end
fprintf('Mean accuracy (OS): %02.2f\n',mean(Acc))
fprintf('Mean accuracy (LR): %02.2f\n',mean(Acc05))

save(output_name ,'y','yhat','Acc','Hyp','Noise','Kf','Alpha')

% check gradients
% % fun   = @(lh)gp_mtr(lh,X,Y,opt);
% % [~,g] = gp_mtr(lh0,X,Y,opt);
% fun   = @(lh)gp_mtr(lh,{Phi, Y},Y,opt);
% [~,g] =gp_mtr(lh0,{Phi, Y},Y,opt);
% gnum  = computeNumericalGradient(fun,lh0);