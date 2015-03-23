%clc
classes = {'ValidHCHitCue',   'ValidHCHitObj',   'ValidLCHitCue',...
           'ValidLCHitObj',   'ValidMissCue',    'ValidMissObj',...
           'InvalidHCHitCue', 'InvalidHCHitObj', 'InvalidLCHitCue',...
           'InvalidLCHitObj', 'InvalidMissCue',  'InvalidMissObj'};
       
prefix{1} = '../gp_da/batch_allpairs/erf500_';
%prefix{1} = '../gp_da/batch_allpairs/t2ml500_';
%prefix{1} = '../gp_da/batch_allpairs/erfmt_';
%prefix{1} = '../gp_da_revisions/loso_batch_allpairs/erfmt_';

prefix{2} = '../gp_da/batch_allpairs/erfnoopt_';
%prefix{2} = '../gp_da/batch_allpairs/noopt_';
%prefix{2} = '../gp_da/batch_allpairs/t2ml500_';
%prefix{2} = '../gp_da/batch_allpairs/single_subject/noopt_';
prefix{2} = '../gp_da/batch_allpairs/single_subject_erf/noopt_';
%prefix{2} = '../gp_da/batch_allpairs/erf500_';
%prefix{2} = '../gp_da_revisions/loso_batch_allpairs/erfmtnoopt_';

C = length(classes);
[c1, c2] = meshgrid(1:length(classes));
c1 = c1.*(tril(ones(C)) - eye(C));
c2 = c2.*(tril(ones(C)) - eye(C));
c1 = c1(:);
c2 = c2(:);
c1 = c1(c1~=0);
c2 = c2(c2~=0);

% collate and plot results  
At2ml = []; Anoopt =[];
for c = 1:length(c1)
    conname = [num2str(c1(c)),'_v_',num2str(c2(c))];
    outname = [prefix{1},conname];
    
    load(outname)
    At2ml  = [At2ml; mean(Acc)];
     
    outname = [prefix{2},conname];
    load(outname)
    Anoopt  = [Anoopt; mean(Acc)]; 
end
id = logical(tril(ones(C)) - eye(C));
D = zeros(length(classes));
Amat_t2ml  = zeros(length(classes));
Amat_noopt = zeros(length(classes));
Amat_t2ml(id)  =  At2ml;
Amat_noopt(id) =  Anoopt;
D(id) = At2ml - Anoopt;
d = D(id);
a = Amat_noopt(id); 
A = Amat_noopt;
A2 = Amat_t2ml;

disp(' ---- all ------');
N_increasing  = sum(d > 0 )
N_decreasing  = sum(d < 0 )
mean_increase = mean(d(d > 0 ))
mean_decrease = mean(d(d < 0))
p_signrank = signrank(At2ml,a)
% disp(' ---- above chance ------');
% N_increasing  = sum(d > 0 & a > 0.5444)
% N_decreasing  = sum(d < 0 & a > 0.5444)
% mean_increase = mean(d(d > 0 & a > 0.5444))
% mean_decrease = mean(d(d < 0 & a > 0.5444))

A = A + A';
A2 = A2 + A2';
D = D + D';
id = [1:2:12, 2:2:12];
A = A(id,id);
A2 = A2(id,id);
D = D(id,id);
A = A - tril(A)';
A(A == 0) = 0.5;
A2 = A2 - tril(A2)';
A2(A2 == 0) = 0.5;
D = D - tril(D)';

%A
%return

subplot(2,2,1)
imagesc(A, [0 1]); colorbar; %colormap(hot)
title('Accuracy of baseline model')
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])

subplot(2,2,2)
imagesc(D, [-0.08 0.08]); colorbar; %colormap(hot)
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])
title('MTL - Baseline');

subplot(2,2,3)
imagesc(A > 0.6); colorbar; %colormap(hot)
%imagesc(A > 0.5444); colorbar; %colormap(hot)
%imagesc(A > 0.6225); colorbar; %colormap(hot)
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])
title('Baseline > chance')

subplot(2,2,4)
imagesc(A2 > 0.6); colorbar; %colormap(hot)
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])
title('t2ml > chance')
