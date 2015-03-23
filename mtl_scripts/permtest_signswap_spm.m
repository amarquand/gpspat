root = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';

c1 = [2, 4];
c2 = [8,10];

[Xa,Ya,ID,~,classes] = load_data;
Ya = [sum(Ya(:,c1),2), sum(Ya(:,c2),2)]; %Ya = Ya(:,clsnum);
id = (1:size(ID,1))';
id = id & sum(Ya,2) > 0;
Ya = Ya(id,:);
Xa = Xa(id,:);
ID = ID(id,:);
[X,y,ID,Y,M] = process_tasks(Xa,Ya,ID);
y(y == 0) = -1;
y(y > 0)  = 1;

[N, T]        = size(Y);

%name  = [root,'gp_da_revisions/weights/spm/ValidHitObj_InvalidHitObj'];

mname = [root,'masks/SPM_mask_46x55x39.img'];
Nm = nifti(mname);
dm = Nm.dat.dim;
mask = reshape(Nm.dat(:,:,:),prod(dm(1:3)),1);
mask(isnan(mask)) = 0;
mask = logical(mask);

Nfold = 10;
Nsub  = 18;
Nperm = 1000;

fprintf('Standardising features ...\n')
Xz = (X - repmat(mean(X),N,1)) ./ repmat(std(X),N,1);
Xz = Xz(:,logical(sum(isfinite(Xz))));
        
[~,~,~,stats] = ttest2(Xz(y == 1,:), Xz(y ~= 1,:));
true_t = stats.tstat;

% single subject
count = zeros(size(true_t));
permname = [root,'gp_da_revisions/weights/spm_perm1000'];
ct10 = 1; p10 = 1;
for p = 1:Nperm
    fprintf('Permutation %d\n',p)
    
    permvect = 2*(rand(size(Xz,1),1) > 0.5) -1;
    Xzp = Xz .* repmat(permvect,1,size(Xz,2));

    % permutation   
    [~,~,~,stats] = ttest2(Xzp(y == 1,:),Xzp(y == -1,:));
    
    perm_t = stats.tstat;
    
    count = count + (abs(true_t) > abs (perm_t));
end

pval = 1- (count ./ Nperm);
sig_voxels = sum(pval < 0.001) %./ length(pval)

prt_write_nii(1-pval,mname,[permname,'_Permtest_pval.img']);
prt_write_nii(true_t,mname,[permname,'_Permtest_tstat.img']);


