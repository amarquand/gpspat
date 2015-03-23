function gp_da_batch_primarycon(seed,cov,machine)

try seed;    catch c = clock; seed = 100*c(6); end
try cov;     catch cov = 'lin'; end
try machine; catch machine = 'cns'; end

%opt.data =data;
opt.optimiseTheta = false;
opt.computeWeights = true;
opt.saveResults    = true;

conname = 'ValidHitObj_InvalidHitObj';
c1 = [2 8];
c2 = [8 10];

% conname = 'ValidCue_ValidObj'
% c1 = [1 3];
% c2 = [2 4];

% conname = 'HCHObj_MissObj';
% c1 = [2 8];
% c2 = [6 12];

nsub = 18;

A = []; A05 = [];
for s = 17:17%1:nsub
    s
    opt.selectTasks = s;
    %outname = ['../gp_da/single_subject/erfnoopt_sub',num2str(opt.selectTasks),'_',conname];
    outname = ['../gp_da_revisions/single_subject/erfnoopt_sub',num2str(opt.selectTasks),'_',conname];
    
%     %[Acc,data, Acc05] = gp_da_run(c1,c2 ,'lin',outname,opt);
%     %[Acc,data, Acc05] = TMP_gp_erf_run(c1,c2 ,'lin',outname,opt)
%     [Acc,data, Acc05] = gp_erf_run_ss(c1,c2 ,'lin',outname,opt);
%     opt.data = data;
%     A = [A; mean(Acc)];
%     A05 = [A05; mean(Acc05)];  

    outname = ['../gp_da_revisions/single_subject/perm',num2str(seed),'_erfnoopt_sub',num2str(opt.selectTasks),'_',conname];
    % permutation test
    %opt.maxEval = 5 %500;
    opt.Nperms = 100;
    opt.rSeed  = seed;
    gp_erf_permtest_ss_weights_retrain(c1,c2 ,'lin',outname,opt);
end

overall = mean(A)
overall05 = mean(A05)
