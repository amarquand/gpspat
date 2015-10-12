function [Yhat, S2] = sp_gpr_gmtl_ols(X,Y,opt)

% sp_gpr_gmtl_ols
%
% this function runs a group spatial analysis on the data defined by the
% input arguments. These are:
%   X    : Nvox x D matrix of spatial locations
%   Y    : Nvox x Ntask x Nsubject matrix of spatial responses
%
% options:
%   opt.Z       : Nsub x Nbeta matrix of subject effects
%   opt.outfile : filename prefix to write incrementally write the output to
%   opt.cov     : gp covariance function
%   opt.mean    : gp mean function
%   opt.lik     : gp likelihood function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% process options and parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Nvox, Nmod, Nsub] = size(Y);
%Nbeta        = size(Z,2);
Nfold        = 2;

% gp paramters 
opt.maxEval  = 200;
opt.inf      = @infExact;
%opt.inf      = @infGrid;
opt.lik      = @likGauss;     opt.hyp0.lik  = log(0.1);

%opt.cov  = {@covMaterniso 5};     opt.hyp0.cov  = zeros(2,1);
% MTL covariance
M = {}; for ii = 1:size(Y,2), M{ii} = ones(Nvox,1); end; M = blkdiag(M{:});
T = size(M,2); l0 = 0.5*eye(T) + 0.5*ones(T); l0 = l0(tril(ones(T)) ~= 0); 
opt.cov = {@sp_covMTL, {@covMaterniso 5}, M}; opt.hyp0.cov = [l0; zeros(2,1)];
sp_covMTL('init',length(opt.hyp0.cov));

%opt.mean = @meanConst;    opt.hyp0.mean = 0;
%opt.mean = {@meanPoly 2}; opt.hyp0.mean = [1;1;1;0;0;0];
opt.mean = @meanZero;    opt.hyp0.mean = [];

opt.usecluster = true;
opt.type2ml    = false;

% generate output filename if required
if isfield(opt,'outfile')
    if iscell(opt.cov), covname = func2str(opt.cov{1});
        if length(opt.cov) > 1 && isnumeric (opt.cov{2}), covname = [covname,num2str(opt.cov{2})]; end
    else covname = func2str(opt.cov);
    end
    if iscell(opt.mean),
        meanname = func2str(opt.mean{1});
        if length(opt.mean) > 1 && isnumeric (opt.mean{2}), meanname = [meanname,num2str(opt.mean{2})]; end
    else meanname = func2str(opt.mean);
    end
    ofile = [opt.outfile,covname,'_',meanname];
end

%%%%%%%%%%%%%%%%%%%%%%%%
% Begin cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%
% reshape into 'Kronecker' style
X = repmat(X,size(M,2),1);
y = reshape(Y,size(Y,1)*size(Y,2),size(Y,3));
if isfield(opt,'Z'), Z = opt.Z; repmat(opt.Z,size(M,2),1); end

% retain only 30% of samples (for speed)
tmp = randperm(length(y)); tmp = tmp(1:floor(length(y)/3));
trvox = false(length(y),1); trvox(tmp) = true; trvox = reshape(trvox,Nvox,T);

stats = struct;
sidxp = randperm(Nsub);
yhat = zeros(size(y)); s2 = zeros(size(y));
for f = 1:Nfold
    fprintf('Fold %d of %d ...\n',f,Nfold);
    [trsub,tesub] = cvindex(Nsub,f,Nfold);
    trsub  = sort(sidxp(trsub)); tesub = sort(sidxp(tesub));
    trl = false(Nsub,1);  trl(trsub) = true;
    tel = false(Nsub,1);  tel(tesub) = true;
          
    % OLS fit
    if isfield(opt,'Z')
        fprintf('Fitting using OLS ... ');
        Btr = zeros(size(Z,2),Nvox,Nmod); Yols = zeros(size(Y));
        
        % find subjects to fit with (accommodating missing data)
        sid = sum(Z ~= 0,2) == size(Z,2) & trl;
        pZZ = pinv(Z(sid,:));
        for m = 1:Nmod
            for v = 1:Nvox
                % fit using training data
                Btr(:,v,m) = pZZ*squeeze(Y(v,m,sid));
                
                % apply to all subjects
                Yols(v,m,:) = (Z*Btr(:,v,m))';
            end
        end
        fprintf('done.\n')
        Yf = Y - Yols;
    else 
        Yf = Y;
    end
    yf = reshape(Yf,size(Y,1)*size(Y,2),size(Y,3));
    
    tic
    opt.cov = {@sp_covMTL, {@covMaterniso 5}, M(trvox,:)}; opt.hyp0.cov = [l0; zeros(2,1)];
    [hyp,nlml] = minimize(opt.hyp0, @sp_gp_sum, opt.maxEval, X(trvox,:), yf(trvox,trsub), opt);
    
    %[nlml,~,hyp] = sp_gp_cluster_batch(opt.hyp0, X(trvox,:), y(trvox,trsub), opt);
    %hyp = opt.hyp0
    toc
    
    % test with all voxels (different subjects) for all tasks except
    % the last. For the last task we fully extrapolate
    predvox = trvox; predvox(:,end) = false;
      
    opt.debug      = true;
    if strcmp(func2str(opt.inf),'infGrid')
        error('not implemented yet')
        %for s = 1:length(tesub)
        %    post = sp_infGrid(hyp, {opt.mean}, opt.cov, opt.lik, X, y(:,tesub(s))); post.L = @(a) a;
        %    [yh,s2] = gp(hyp, @sp_infGrid, opt.mean, opt.cov, opt.lik, Xtr, post, X); 
        %end
        %Yhat(:,tesub(s)) = yh;
        %S2(:,tesub(s))   = s2;
    else
        opt.cov = {@sp_covMTL, {@covMaterniso 5}, {M(predvox(:),:) M}};
        [~,~,~,yh,s2f] = sp_gp_cluster_job(hyp,X(predvox,:),y(predvox,tesub),opt,X); 
        yhat(:,tesub)  = yh;
        s2(:,tesub)    = s2f;
    end
    opt.debug      = false;
    
    Yhat = reshape(yhat,size(Y));
    S2   = reshape(s2,size(Y));
    % add OLS predictions back in
    if isfield(opt,'Z'), Yhat = Yhat + Yols;  end
    
    disp('Computing stats...');
    SE   = (yhat(:,tel) - y(:,tel)).^2;
    MSE  = mean(reshape(SE,size(Y,1),size(Y,2),sum(tel)));
    %MSE  = mean((Y(:,:,tel)-Yhat(:,:,tel)).^2);
    SMSE = MSE ./var(Y(:,:,tel));
    fprintf('Done. mean (std) SMSE = %2.2f (%2.2f) \n',mean(SMSE(:)),std(SMSE(:)));
    
    stats(f).trvox = trvox; 
    stats(f).trsub = trsub; 
    stats(f).predvox = predvox; 
    stats(f).M = M;
    stats(f).opt = opt; 
    %stats(f).y = y; 
    stats(f).yhat = yhat; 
    stats(f).S2 = S2; 
    stats(f).NLML = min(nlml); 
    stats(f).HYP = hyp;
    stats(f).SE = SE; 
    stats(f).MSE = MSE; 
    stats(f).SMSE = SMSE;
    
    if isfield(opt,'outfile'), save(ofile,'stats'); end
end