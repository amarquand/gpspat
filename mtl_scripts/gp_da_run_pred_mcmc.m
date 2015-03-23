clear;

OutputFilename = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da/mcmc_24_v_810_ValidHitObj_InvalidHitObj_eye_prior/'; % leave blank for no output
c1 = [2 4];
c2 = [8 10];

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configure MCMC parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
yhattr = cell(Nfolds,1);
yhatte = cell(Nfolds,1);
s2te   = cell(Nfolds,1);
Noise  = zeros(T,Nfolds);
Kf     = zeros(T,T,Nfolds);
matlabpool('open');
parfor f = 1:Nfolds
    fprintf('Outer loop %d of %d ...\n',f,Nfolds)

    sid = ID(:,2) == f;
    if sum(sid) == 0, error(['No tasks found for fmri run ', num2str(r)]); end
    te = find(sid);
    tr = find(~sid);
    teall(:,f) = sid;
    trall(:,f) = ~sid;
        
    fstats = load ([OutputFilename,'fold_',num2str(f),'_stats']); stats = fstats.stats;
    optf = stats.opt;
    optf.OutputFilename = [OutputFilename,'fold_',num2str(f),'_'];
    optf.TestInterval = 1;
    optf.BurnIn = 5000;
    
    %fprintf('Standardising features ...\n ')
    Xz  = (X - repmat(mean(X(tr,:)),N,1)) ./ repmat(std(X(tr,:)),N,1);
    Xz  = Xz(:,logical(sum(isfinite(Xz))));
    Phi = Xz*Xz'; 
    if strcmp(optf.CovFunc,'covfunc_mtr_se') || strcmp(optf.CovFunc,'covfunc_mtr_se2')
        disp('Normalizing kernel ...');
        Phi = prt_normalise_kernel(Phi);
    end
     
      
    [yhatte{f}, Noise(:,f), Kf(:,:,f), s2te{f}] = gp_pred_mtr_mh_nonblock({Phi, Ys},tr,te,{y(tr)},optf);
        
    %Yhat(f,:) = Yhat(f,:) + mtr;
    %S2(f,:)   = s2te{f};
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

save([OutputFilename,'results_all'],'y','yhat','S2','Noise','Acc','Acc05')


