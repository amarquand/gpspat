classes = {'ValidHCHitCue',   'ValidHCHitObj',   'ValidLCHitCue',...
           'ValidLCHitObj',   'ValidMissCue',    'ValidMissObj',...
           'InvalidHCHitCue', 'InvalidHCHitObj', 'InvalidLCHitCue',...
           'InvalidLCHitObj', 'InvalidMissCue',  'InvalidMissObj'};
       
C = length(classes);
[c1, c2] = meshgrid(1:length(classes));
c1 = c1.*(tril(ones(C)) - eye(C));
c2 = c2.*(tril(ones(C)) - eye(C));
c1 = c1(:);
c2 = c2(:);
c1 = c1(c1~=0);
c2 = c2(c2~=0);

opt.optimiseTheta = true
opt.saveResults   = false;
opt.computeWeights = false;
opt.maxEval       = 500;

% run analyses
A = []; A05 = [];
for c = 1:length(c1)
    disp('---------------------------------- ');
    c
    disp('---------------------------------- ');
    
    conname = [num2str(c1(c)),'_v_',num2str(c2(c))];
    outname = ['../gp_da/batch_allpairs/erf500_',conname];
    %outname = ['../gp_da/batch_allpairs/erfnoopt_',conname];
    %outname = ['../gp_da/batch_allpairs/erfmt_',conname];
    %outname = ['../gp_da/batch_allpairs/erf500mt00_',conname];
    %outname = ['../gp_da_revisions/loso_batch_allpairs/erfmtnoopt_',conname];

    %[Acc,data, Acc05] = gp_da_run(c1(c),c2(c) ,'lin',outname,opt);
    [Acc,data, Acc05] = gp_erf_run(c1(c),c2(c) ,'lin',outname,opt);
    %[Acc,data, Acc05] = gp_erf_run(c1(c),c2(c) ,'linmt',outname,opt);
    %[Acc,data, Acc05] = gp_erf_run_loso(c1(c),c2(c) ,'linmt',outname,opt);
    opt.data = data;

    A = [A; mean(Acc)];
    A05 = [A05; mean(Acc05)];
    save(outname,'Acc','Acc05');
end

overall = mean(A)
overall05 = mean(A05)

% % collate and plot results  
% At2ml = []; Anoopt =[];
% for c = 1:length(c1)
%     conname = [num2str(c1(c)),'_v_',num2str(c2(c))];
%     outname = ['../gp_da/batch_allpairs/erf500_',conname];
%     %outname = ['../gp_da/batch_allpairs/t2ml500_',conname];
%     load(outname)
%     %auc    = roc(y,yhat);
%     At2ml  = [At2ml; mean(Acc)];
%     %At2ml  = [At2ml; auc];
%     
%     %outname = ['../gp_da/batch_allpairs/erfnoopt_',conname];
%     %outname = ['../gp_da/batch_allpairs/noopt_',conname];
%     outname = ['../gp_da/batch_allpairs/t2ml500_',conname];
%     load(outname)
%     %auc     = roc(y,yhat);
%     Anoopt  = [Anoopt; mean(Acc)]; 
%     %Anoopt  = [Anoopt; auc]; 
% end
% id = logical(tril(ones(C)) - eye(C));
% D = zeros(length(classes));
% Amat_t2ml  = zeros(length(classes));
% Amat_noopt = zeros(length(classes));
% Amat_t2ml(id)  =  At2ml;
% Amat_noopt(id) =  Anoopt;
% D(id) = At2ml - Anoopt;
% d = D(id);
% a = Amat_noopt(id);
% A = Amat_noopt;
% 
% N_increasing  = sum(d > 0 & a > 0.5444)
% N_decreasing  = sum(d < 0 & a > 0.5444)
% mean_increase = mean(d(d > 0 & a > 0.5444))
% mean_decrease = mean(d(d < 0 & a > 0.5444))
% 
% A = A + A';
% D = D + D';
% id = [1:2:12, 2:2:12];
% A = A(id,id);
% D = D(id,id);
% A = A - tril(A)';
% A(A == 0) = 0.5;
% D = D - tril(D)';
% 
% subplot(1,3,1)
% imagesc(A, [0 1]); colorbar; %colormap(hot)
% subplot(1,3,2)
% imagesc(D, [-0.075 0.075]); colorbar; %colormap(hot)
% subplot(1,3,3)
% imagesc(A > 0.5444); colorbar; %colormap(hot)
% 
% 
