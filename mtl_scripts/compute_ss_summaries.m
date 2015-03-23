dir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da/single_subject/';

nsub = 18;

con = {'ValidCue_ValidObj','ValidHitObj_InvalidHitObj','HCHObj_MissObj'};

for c = 1:length(con)
    A = [];
    for s = 1:nsub
        load([dir,'noopt_sub',num2str(s),'_',con{c}]);
        A = [A; Acc];
    end
    Acc = A;
    save([dir,'noopt_',con{c}]);
end
%erfnoopt_sub18_HCHObj_MissObj.mat         
%noopt_sub18_HCHObj_MissObj.mat
%erfnoopt_sub18_ValidCue_ValidObj.mat  
%noopt_sub18_ValidCue_ValidObj.mat
%erfnoopt_sub18_ValidHitObj_InvalidHitObj.mat 
%noopt_sub18_ValidHitObj_InvalidHitObj.mat
