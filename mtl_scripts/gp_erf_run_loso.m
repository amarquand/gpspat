function [Acc, data, Acc05] = gp_erf_run(c1,c2,cov,output_name,opt)

% %clsnum = [1,2];
% c1  = 1; c2 =2; 
% cov = 'lin';
% output_name = ['../gp_da/t2ml_',mat2str(c1),'_v_',num2str(mat2str(c2)),'_YY0'];
% opt.maxEval = 500;
% opt.optimiseTheta = false;

addpath(genpath('/home/kkvi0203/sfw/pronto_dev'))
addpath('/software/system/spm/spm-8-5236/')
addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');

% defaults
try opt.optimiseTheta;  catch, opt.optimiseTheta = true;   end
try opt.computeWeights; catch, opt.computeWeights = false; end
try opt.normalizeK;     catch, opt.normalizeK = false;     end
try opt.maxEval;        catch, opt.maxEval = 500;          end
try opt.saveResults;    catch, opt.saveResults = true;     end

% get data
if isfield(opt,'data')
    disp('Using existing data matrices ...');
    Xa = opt.data.Xa;
    Ya = opt.data.Ya;
    ID = opt.data.ID;
else
    disp('Loading data ...');
    [Xa,Ya,ID,~,classes] = load_data;
end
data.Xa = Xa;
data.Ya = Ya;
data.ID = ID;

if isfield(opt,'selectTasks')
    disp(['selecting tasks: ',mat2str(opt.selectTasks)])
    id = ID(:,1) == opt.selectTasks;
else
    id = (1:size(ID,1))';
end

% get rid of fMRI tasks we are not considering
Ya = [sum(Ya(:,c1),2), sum(Ya(:,c2),2)]; %Ya = Ya(:,clsnum);
id = id & sum(Ya,2) > 0;
Ya = Ya(id,:);
Xa = Xa(id,:);
ID = ID(id,:);

[X,y,ID,Y,M] = process_tasks(Xa,Ya,ID);
[N, T]        = size(Y);
Nclassifiers  = length(unique(ID(:,1)));
Nfolds        = length(unique(ID(:,1)));

% starting hyperparamers
Kf00  = eye(T);
%lam   = 0.5;%1/(1+exp(-0));
%Kf00  = (1-lam)*ones(T) + lam*eye(T);
kf00 = Kf00(tril(ones(T)) ~= 0);

% options
switch cov
    case 'se'
        error('Squared exponential not adjusted yet');
        opt.CovFunc = 'covfunc_mtr_se'; kx0 = [log(1); log(1000); log(0.1*ones(T,1))];
        kx0 = [log(1); log(1); log(0.1*ones(T,1))];
        lh00 = [kf00; kx0];
    case 'linmt' 
        opt.CovFunc = 'covfunc_mtr_nonblock_meantask'; % dummy value    
        covfunc   = @covLINmtlmeantask;
        kx0 =[];
        lh00 = 0;  % moderately coupled
        %lh00 = 10;  % uncoupled
    otherwise % linear
        opt.CovFunc = 'covfunc_mtr_nonblock'; kx0 = log(0.1*ones(T,1));
        %%opt.CovFunc = 'covfunc_mtr'; kx0 = log(ones(T,1));
        lh00 = [kf00; kx0];
        
        covfunc   = @covLINmtl;
end

% GPML stuff
%global Nyp; 
Nhyp      = length(lh00);
meanfunc  = @meanZeromtl;
likfunc   = @likErf;
inffunc   = @prt_infEP;
y(y == 0) = -1;
y(y > 0)  = 1;
hyp0.cov  = lh00;

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
    
    sid = ID(:,1) == f;
    if sum(sid) == 0, 
        disp(['No tasks found for subject ', num2str(f)]);
        yhatte{f} = NaN;
        bad = true;
    else
        bad = false;
    end
    te = find(sid);
    tr = find(~sid);
    teall(:,f) = sid;
    trall(:,f) = ~sid;
       
    % selection matrices
    Mtr = sparse(M(tr,:));
    Mte = sparse(M(te,:));
  
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
        %%Kf0 = eye(T);
        %Kf0 = pinv(full(Mtr))*(y(tr)*y(tr)' ./ Phi(tr,tr))*pinv(full(Mtr))'; optf.Psi_prior = Kf0;
        %kf0 = Kf0(tril(ones(T)) ~= 0);
        %lh0 = [kf0; kx0]; hyp0.cov = lh0;
        
        hyp = minimize(hyp0, @prt_gp, opt.maxEval, inffunc, meanfunc, ...
                       covfunc, likfunc, {Phi(tr,tr), Mtr}, y(tr));
        %[hyp nlmls] = minimize(lh0, @gp_mtr, opt.maxEval, {Phi(tr,tr), Mtr}, {y(tr)}, opt);
    else
        %Kf0     = eye(T);
        %kf0     = Kf0(tril(ones(T)) ~= 0);
        kf0     = 10;
        hyp.cov = [kf0; kx0];
    end
        
    if ~bad
        [ymu ys2 fmu fs2 lp post] = ...
            prt_gp(hyp,inffunc,meanfunc,covfunc,likfunc, ...
                   {Phi(tr,tr), Mtr}, y(tr), ...
                   {Phi(te,tr), Mte, Mtr}, zeros(length(te),1), ...
                   {Phi(te,te), Mte, Mte});
        yhatte{f} = exp(lp);  
        alpha = post.alpha;
    end
   
    Hyp(f,:)   = hyp.cov';
    
    % weights
    if opt.computeWeights && ~bad
        %error('Weights not adjusted yet');
        disp('Computing weights ...');
        mask        = '/cns_zfs/mlearn/public_datasets/openfmri/posner/masks/SPM_mask_46x55x39.img';
        %[~,~,alpha] = gp_mtr(hyp, {Phi(tr,tr), Mtr}, {y(tr)}, opt);
        nvox        = size(X,2);
        Xtr         = Xz(tr,:);
        
        % generate "mean" weight vector
        Wm  = alpha'*Xtr;
        Wmn = Wm ;%./ norm(Wm); % for visualisation only 
        prt_write_nii(Wmn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_meantask.img']);
        
        % old way
        % W  = zeros(T,nvox);
        % for t = 1:T
        %     xid = find(ID(:,1) == t & ID(:,2) ~= f);
        %     aid = find(ID(tr,1) == t);
        %     
        %     W(t,:) = alpha(aid)'*Xz(xid,:);
        %    
        %     wn = W(t,:);% ./ norm(W(t,:));
        %     prt_write_nii(wn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(t),'.img']);
        % end 
        
        % Reconstruct chol(Kx)' and Kf
        if strcmp(cov,'linmt')
            lam     = 1/(1+exp(-hyp.cov(1)));
            Kf      = (1-lam)*ones(T) + lam*T*eye(T);
        else
            lmaxi = (T*(T+1)/2);
            Lf     = zeros(T);
            lf     = hyp.cov(1:lmaxi);
            id     = tril(true(T));
            Lf(id) = lf;
            Kf     = Lf*Lf';
        end
        
        % compute A and Xtilde
        A    = kron(Kf,speye(nvox));
        Xttr = cell(T,1);
        for t = 1:T
           Xttr{t} = Xtr(Mtr(:,t) ~= 0, :);
        end
        Xttr = sparse(blkdiag(Xttr{:}));   % very inefficient       
        w    = A*Xttr'*alpha;
        W    = reshape(w,nvox,T);
        for t = 1:T
            wn = W(:,t)';% ./ norm(W(t,:));
            prt_write_nii(wn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(t),'.img']);
        end
        %clear A Xtr Xttr W w
    end

    fprintf('Outer loop %d of %d done.\n',f,Nfolds)
end
%matlabpool('close')

% reconstruct predictions
yhat   = zeros(N,1); 
%s2     = zeros(N,1);
for f = 1:Nfolds
   te = find(teall(:,f));
   yhat(te) = yhatte{f};
   %s2(te)   = s2te{f};
end

% Reconstruct chol(Kx)' and Kf
lmaxi = (T*(T+1)/2);
%Noise = zeros(Nfolds,T); 
Kf    = zeros(T,T,Nfolds);
% for f = 1:Nfolds
%     Lf        = zeros(T);
%     lf        = Hyp(f,1:lmaxi)';
%     id        = tril(true(T));
%     Lf(id)    = lf;
%     Kf(:,:,f) = Lf*Lf';
% %    Noise(f,:) = exp(Hyp(f,end-T+1:end));
% end

% compute accuracy
Acc    = zeros(Nclassifiers,1);
Acc05  = zeros(Nclassifiers,1);
for c = 1:Nclassifiers
    %c
    clsid = find(ID(:,4) == c);
    trlab = y(clsid) == 1 ;
    
    Yf = [y(clsid) y(clsid(end:-1:1))];
    
    %prlab    = opt_score(Yf,Yf,yhat(clsid));
    prlab    = yhat(clsid) > 0.5;
    
    % ignore runs for which there was no data
    prlab = prlab(isfinite(prlab));
    trlab = trlab(isfinite(prlab));
    
    Acc(c)   = sum(trlab == prlab) ./ length(trlab);
end
fprintf('Mean accuracy: %02.2f\n',mean(Acc))
%fprintf('Mean accuracy (LR): %02.2f\n',mean(Acc05))

if opt.saveResults
    save(output_name ,'y','yhat','Acc','Hyp','Kf','Alpha','ID')
end

% check gradients
% % fun   = @(lh)gp_mtr(lh,X,Y,opt);
% % [~,g] = gp_mtr(lh0,X,Y,opt);
% fun   = @(lh)gp_mtr(lh,{Phi, Y},Y,opt);
% [~,g] =gp_mtr(lh0,{Phi, Y},Y,opt);
% gnum  = computeNumericalGradient(fun,lh0);