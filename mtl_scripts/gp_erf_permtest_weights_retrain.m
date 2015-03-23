function [] = gp_erf_permtest_weights_retrain(c1,c2,cov,output_name,opt)

RandStream.setGlobalStream(RandStream('mt19937ar','seed',opt.rSeed));

%addpath(genpath('/home/kkvi0203/sfw/pronto_dev'))
%addpath('/software/system/spm/spm-8-5236/')
%addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');

% defaults
try opt.optimiseTheta;  catch, opt.optimiseTheta = true;   end
try opt.computeWeights; catch, opt.computeWeights = false; end
try opt.normalizeK;     catch, opt.normalizeK = false;     end
try opt.maxEval;        catch, opt.maxEval = 500;          end
try opt.saveResults;    catch, opt.saveResults = true;     end
try opt.rootDir;        catch, opt.rootDir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/'; end
    
% get data
if isfield(opt,'data')
    disp('Using existing data matrices ...');
    Xa = opt.data.Xa;
    Ya = opt.data.Ya;
    ID = opt.data.ID;
else
    disp('Loading data ...');
    [Xa,Ya,ID,~,classes] = load_data(opt.rootDir);
end
%data.Xa = Xa;
%data.Ya = Ya;
%data.ID = ID;

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

[X,y,ID,Y,M]  = process_tasks(Xa,Ya,ID);
[N, T]        = size(Y);
Nclassifiers  = length(unique(ID(:,1)));
Nfolds        = length(unique(ID(:,2)));
N_per_fold    = T*(length(c1)+length(c2));
nvox          = size(X,2);

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
    case 'linmt' 
        opt.CovFunc = 'covfunc_mtr_nonblock_meantask'; % dummy value    
        covfunc   = @covLINmtlmeantask;
        kx0 =[];
        lh00 = 0; 
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
pstats.ID        = ID;
pstats.opt       = opt;
pstats.T         = zeros(opt.Nperms,nvox);
pstats.permvects = zeros(N,Nfolds,opt.Nperms);
pstats.yhat      = zeros(N,opt.Nperms);
pstats.hyp       = zeros(Nfolds,length(lh00),opt.Nperms);

%matlabpool('open');
for perm =1:opt.Nperms
    trall  = zeros(N,Nfolds);
    teall  = zeros(N,Nfolds);
    yhatte = cell(Nfolds,1);
    Hyp    = zeros(Nfolds,length(lh00));    
    %XWa    = nan(T*Nfolds*(length(c1)+length(c2)),nvox); xwaoffs = 0;
    
    XWc       = cell(opt.Nperms,1);
    permvects = zeros(N,Nfolds);
    %par
    for f = 1:Nfolds
        fprintf('Outer loop %d of %d ...\n',f,Nfolds)
        
        sid = ID(:,2) == f;
        if sum(sid) == 0,
            disp(['No tasks found for fmri run ', num2str(f)]);
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
        
        fprintf('Standardising features ...\n')
        Xz = (X - repmat(mean(X(tr,:)),N,1)) ./ repmat(std(X(tr,:)),N,1);
        Xz = Xz(:,logical(sum(isfinite(Xz))));
        
        fprintf('Permuting scans ...\n')
        permvects(:,f) = 2*(rand(size(Xz,1),1) > 0.5) -1;
        Xz             = Xz .* repmat(permvects(:,f),1,size(Xz,2));
        
        Phi = Xz*Xz';
        if strcmp(opt.CovFunc,'covfunc_mtr_se') || opt.normalizeK
            disp('Normalizing kernel ...');
            Phi = prt_normalise_kernel(Phi);
        end
        
        if opt.optimiseTheta
            hyp = minimize(hyp0, @prt_gp, opt.maxEval, inffunc, meanfunc, ...
                covfunc, likfunc, {Phi(tr,tr), Mtr}, y(tr));
        else
            hyp = hyp0;
        end
        
        if ~bad
            [ymu, ys2, fmu, fs2, lp, post] = ...
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
            Xtr         = Xz(tr,:);
            
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
            XW   = nan(T*(length(c1)+length(c2)),nvox); xwoffs = 0;
            for t = 1:T
                wn = W(:,t)';% ./ norm(W(t,:));
                xid = find(ID(:,1) == t & ID(:,2) == f); %train
                
                xw    = X(xid,:).*repmat(wn,length(xid),1);
                xwidx = (1:size(xw,1)) + xwoffs; xwoffs = xwoffs+size(xw,1);
                XW(xwidx,:) = xw;
                %prt_write_nii(wn,mask,[output_name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(t),'.img']);
            end
            clear A Xtr Xttr W w
        end
        XWc{f}= XW;
        
        %fprintf('Outer loop %d of %d done.\n',f,Nfolds)
    end
    
    % reassemble permuted data matrix
    XWa    = nan(Nfolds*N_per_fold,nvox); xwaoffs = 0;
    yhat   = zeros(N,1); 
    for f = 1:Nfolds
        %xwaidx   = (1:size(XW,1)) + xwaoffs; xwaoffs = xwaoffs+size(XW,1);
        xwaidx   = (1:N_per_fold) + (f-1)*N_per_fold;
        XWa(xwaidx,:) = XWc{f};

        yhat(find(teall(:,f))) = yhatte{f};
    end
    clear XWc
    % get rid of missing scans
    nz  = isfinite(XWa(:,1));
    XWa = XWa(nz,:);
    
    [~,~,~,stats] = ttest(XWa);
    clear XWa
    
    pstats.T(perm,:)           = stats.tstat;
    pstats.perm                = perm;
    pstats.permvects(:,:,perm) = permvects;
    pstats.yhat(:,perm)        = yhat;
    pstats.hyp(:,:,perm)       = Hyp;
    
    if opt.saveResults
        save(output_name ,'pstats');
    end
end
%matlabpool('close')
