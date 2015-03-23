function gp_da_batch_primarycon(seed,cov,machine)

try seed;    catch c = clock; seed = 100*c(6); end
try cov;     catch cov = 'lin'; end
try machine; catch machine = 'cns'; end

% if strcmp(cov,'linmt') % 'lin'
%     outname = ['../gp_da_revisions/perm',num2str(seed),'_erfmt_ValidHitObj_InvalidHitObj'];
% else
%     outname = ['../gp_da_revisions/perm',num2str(seed),'_erf_ValidHitObj_InvalidHitObj'];
% end

outname = ['../gp_da_revisions/perm',num2str(seed),'_erfnoopt_ValidHitObj_InvalidHitObj'];

if strcmp(machine,'cns') % 'lin'
    opt.rootDir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';
    addpath(genpath('/home/kkvi0203/sfw/pronto_dev'))
    addpath('/software/system/spm/spm-8-5236/')
    addpath('/home/kkvi0203/svmdata/PD_MSA_PSP/prt/prt_mcode/mtl_clean');
else
    opt.rootDir = '/home/andre/cnsdata2/public_datasets/openfmri/posner/';
    addpath(genpath('/home/andre/Dropbox/sfw/pronto/trunk'))
    addpath('/home/andre/sfw/spm8-4667/')
    addpath('/home/andre/cns/sfw/gpmtl');
end
    
opt.optimiseTheta  = false;
opt.saveResults    = true;
opt.computeWeights = true;
opt.maxEval        = 5 %500;
% permutation test
opt.Nperms = 100;
opt.rSeed  = seed;

%gp_erf_run([1 3],[2 4],'linmt','../gp_da_revisions/erfmt_ValidCue_ValidObj',opt)
%gp_erf_run([2, 4], [8,10],'linmt','../gp_da_revisions/erfmt_ValidHitObj_InvalidHitObj',opt)
%gp_erf_run([2, 8], [6,12],'linmt','../gp_da_revisions/erfmt_HCHObj_MissObj',opt)


gp_erf_permtest_weights_retrain([2, 4], [8,10],cov,outname,opt)