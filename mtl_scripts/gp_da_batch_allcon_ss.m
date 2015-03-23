classes = {'ValidHCHitCue',   'ValidHCHitObj',   'ValidLCHitCue',...
           'ValidLCHitObj',   'ValidMissCue',    'ValidMissObj',...
           'InvalidHCHitCue', 'InvalidHCHitObj', 'InvalidLCHitCue',...
           'InvalidLCHitObj', 'InvalidMissCue',  'InvalidMissObj'};
      
nsub = 18;
C = length(classes);
[c1, c2] = meshgrid(1:length(classes));
c1 = c1.*(tril(ones(C)) - eye(C));
c2 = c2.*(tril(ones(C)) - eye(C));
c1 = c1(:);
c2 = c2(:);
c1 = c1(c1~=0);
c2 = c2(c2~=0);

opt.optimiseTheta = false;
opt.saveResults   = false;
opt.maxEval       = 500;

for c = 1:length(c1)
    disp('---------------------------------- ');
    c
    disp('---------------------------------- ');
    conname = [num2str(c1(c)),'_v_',num2str(c2(c))];
    outname = ['../gp_da/batch_allpairs/single_subject_erf/noopt_',conname];
        
    A = []; A05 = [];
    for s = 1:nsub
        s
        opt.selectTasks = s;
        
        %[Acc,data, Acc05] = gp_da_run(c1(c),c2(c) ,'lin',outname,opt);
        [Acc,data, Acc05] = TMP_gp_erf_run(c1(c),c2(c) ,'lin',outname,opt);
        opt.data = data;
        
        A = [A; mean(Acc)];
        A05 = [A05; mean(Acc05)];
    end
    Acc = A; Acc05 = A05;
    save(outname,'Acc','Acc05');
end

overall = mean(A)
overall05 = mean(A05)

% % collate results  
% At2ml = []; Anoopt =[];
% for c = 1:length(c1)
%   
%     outname = ['../gp_da/batch_allpairs/t2ml500_',conname];
%     load(outname)
%     At2ml  = [At2ml; mean(Acc)];
%     outname = ['../gp_da/batch_allpairs/noopt_',conname];
%     load(outname)
%     Anoopt  = [Anoopt; mean(Acc)];  
% end
% id = logical(tril(ones(C)) - eye(C));
% D = zeros(length(classes));
% Amat_t2ml  = zeros(length(classes));
% Amat_noopt = zeros(length(classes));
% Amat_t2ml(id)  =  At2ml;
% Amat_noopt(id) =  Anoopt;
% D(id) = At2ml - Anoopt;