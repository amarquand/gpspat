%wdir = '/home/andre/cnsdata2/public_datasets/openfmri/posner/gp_da_revisions/';
wdir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da_revisions/';

cov = 'erfnoopt'

%normal
names = dir([wdir,'*',cov,'*']);

% ct = 0;
% permT = [];
% for n = 1:length(names)
%    load(fullfile(wdir,names(n).name));
%    ct = ct + pstats.perm;
%    permT = [permT; pstats.T(1:pstats.perm,:)];
%    if size(permT,1) >= 1000
%        break
%    end
%    %pstats.perm
% end
% ct
%
% permT = permT(1:1000,:);
% save ([wdir,cov,'_perm1000'],'permT')

wdir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da_revisions/single_subject/';
N_per_fold    = 4;
Nfolds        = 10;
Nsubjects     = 18;
for p10 = 1:10
    W10all = cell(Nsubjects,1);
    for s = 1:Nsubjects
        disp(['part ',num2str(p10),' loading subject ',num2str(s)])
        load([wdir,'perm',num2str(p10),'_erfnoopt_sub',num2str(s),'_ValidHitObj_InvalidHitObj'])
        
        W10all{s} = pstats.XW;
        clear pstats;
    end
    
    W = cell(100,1);
    for pp2 = 1:100%pstats.perm
        pp2
        Wp = cell(Nsubjects,1);
        for s = 1:Nsubjects            
            %Wp{s} = pstats.XW{pp2};
            Wp{s} = W10all{s}{pp2};
        end
        W{pp2} = cell2mat(Wp);
    end
    save([wdir,'erfnoopt_perm1000_',num2str(p10)],'W','-v7.3');
    clear W10all W Wp
end


