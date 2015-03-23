root = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';

%name  = [root,'gp_da/weights/erf_ValidHitObj_InvalidHitObj/erf_ValidHitObj_InvalidHitObj']
%name  = [root,'gp_da/weights/erfmt_ValidHitObj_InvalidHitObj/erfmt_ValidHitObj_InvalidHitObj']
%name  = [root,'gp_da/single_subject/weights/erfnoopt_ValidHitObj_InvalidHitObj/erfnoopt_ValidHitObj_InvalidHitObj']
name  = [root,'gp_da/weights/erfnoopt_ValidHitObj_InvalidHitObj/erfnoopt_ValidHitObj_InvalidHitObj'];

mname = [root,'masks/SPM_mask_46x55x39.img'];
Nm = nifti(mname);
dm = Nm.dat.dim;
mask = reshape(Nm.dat(:,:,:),prod(dm(1:3)),1);
mask(isnan(mask)) = 0;
mask = logical(mask);

Nfold = 10;
Nsub  = 18;
Nperm = 1000;

W = [];
for s = 1:Nsub
    
    fprintf('Loading subject %d\n',s)
    for f = 1:Nfold
       %if f ~= 8
       try
%             N = nifti([name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(s),'.img']);
%             w = N.dat(:,:,:);
%             w = w(:);% ./ norm(w(:));
%             w = w(mask);
%             %w = w ./ max(abs(w));
            
            N = nifti([name,'_XW_fold_',num2str(f,'%02.0f'),'_task',num2str(s),'.img']);
            w = [];
            for i = 1:N.dat.dim(4)
                wi = N.dat(:,:,:,i);
                wi = wi(:);  
                wi = wi(mask);
                w = [w wi];
            end
             W = [W; w'];
       catch
           fprintf('couldn''t read subject %d fold %d\n',s,f)
       end
       %end
    end
end
%W = W';

[~,~,~,stats] = ttest(W);
true_t = stats.tstat;

% old way (permute voxels independently)
% count = zeros(size(true_t));
% for p = 1:Nperm
%     %if (~mod(p,50)), fprintf('Permutation %d\n',p); end
%     fprintf('Permutation %d\n',p)
%     % permutation
%     P  = 2*(rand(size(W)) > 0.5) -1;
%     Wp = W .* P;
%     
%     [~,~,~,stats] = ttest(Wp);
%     
%     perm_t = stats.tstat;
%     
%     count = count + (abs(true_t) > abs (perm_t));
% end

% %new way (multi-subject)
% permname = [root,'gp_da_revisions/erfnoopt_perm1000'];
% load(permname)
% count = sum(repmat(abs(true_t),size(permT,1),1) > abs (permT));

% single subject
count = zeros(size(true_t));
permname = [root,'gp_da_revisions/single_subject/erfnoopt_perm1000'];
ct10 = 1; p10 = 1;
for p = 1:Nperm
    if ~mod(p-1,100)
        fprintf('Loading ... %d\n',p); 
        load([permname,'_',num2str(ct10)]);
        ct10 = ct10+1;
        p10  = 1;
    end
    
    fprintf('Permutation %d\n',p)
    % permutation   
    [~,~,~,stats] = ttest(W{p10});
    
    perm_t = stats.tstat;
    
    count = count + (abs(true_t) > abs (perm_t));
    p10 = p10 + 1;
end

pval = 1- (count ./ Nperm);
sig_voxels = sum(pval < 0.001) %./ length(pval)

prt_write_nii(1-pval,mname,[permname,'_Permtest_pval.img']);
prt_write_nii(true_t,mname,[permname,'_Permtest_tstat.img']);


