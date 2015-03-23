function [Acc] = gp_da_run_mcmc(c1,c2,cov,output_name,opt)

RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));
addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');

[Xa,Ya,ID,mask,classes] = load_data;

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

%%%%%%%%%%%%%%%%%%%%%%%%
% starting hyperparamers
%%%%%%%%%%%%%%%%%%%%%%%%
Kf0 = 1e-3*eye(T);
Tm  = tril(ones(T));
kf0 = Kf0(Tm ~= 0);
lh0 = [kf0; log(0.1*ones(T,1))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configure MCMC parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt.nGibbsIter     = 20000;
opt.BurnIn         = 1000;
opt.TestInterval   = 100;
opt.WriteInterim   = true;
opt.mh.StepSize      = 0.1;%0.25;
opt.mh.ProposalScale = 5000; %200;
   
% Linear covariance
opt.CovFunc        = 'covfunc_mtr_nonblock'; kx0 = [log(0.1*ones(T,1))];
opt.UseYYPrior     = true;
if opt.UseYYPrior
    opt.PriorParam     = [1,T+2,1,3]; % pi = IW(Kf|Kfhat,T+2)*prod(IG(s2|1,3))
else
    opt.PriorParam     = [1e-5,T,1,3];
end

opt.OutputFilename = [output_name,'/']; % leave blank for no output
opt.X0_MH          = lh0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Cross-validation loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mkdir(opt.OutputFilename);

trall  = zeros(N,Nfolds);
teall  = zeros(N,Nfolds);
matlabpool('open');
parfor f = 1:Nfolds
    fprintf('Outer loop %d of %d ...\n',f,Nfolds)

    sid = ID(:,2) == f;
    if sum(sid) == 0, error(['No tasks found for fmri run ', num2str(r)]); end
    te = find(sid);
    tr = find(~sid);
    teall(:,f) = sid;
    trall(:,f) = ~sid;
           
    Xz = (X - repmat(mean(X(tr,:)),N,1)) ./ repmat(std(X(tr,:)),N,1);
    Xz = Xz(:,logical(sum(isfinite(Xz))));
    Phi = Xz*Xz';
    if strcmp(opt.CovFunc,'covfunc_mtr_se') || strcmp(opt.CovFunc,'covfunc_mtr_se2')
        disp('Normalizing kernel ...');
        Phi = prt_normalise_kernel(Phi);
    end
    
    % training mean
    Ystr = Ys(tr,:);
    %Ytr = Y(tr,:);
    %mtr = mean(Y(tr,:));
    %Mtr = repmat(mtr,length(tr),1);
    %Ytr = Ytr - Mtr;
    
    optf = opt;
    optf.OutputFilename = [optf.OutputFilename,'fold_',num2str(f),'_'];
    optf.X0_MH = lh0;
    
    if strcmp(opt.CovFunc,'covfunc_mtr_se') || strcmp(opt.CovFunc,'covfunc_mtr_se2')
        gp_mtr_se_gibbs_mh({Phi(tr,tr), Ystr}, {y(tr)}, optf);
    else
        %gp_mtr_mh({Phi(tr,tr), Ytr}, Ytr, optf);
        gp_mtr_gibbs_mh({Phi(tr,tr), Ystr}, {y(tr)}, optf);
        %gp_mtr_gibbs_mh_chol({Phi(tr,tr), Ytr}, Ytr, optf);
    end
end
matlabpool close
