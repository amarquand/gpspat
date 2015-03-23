wdir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da/';


fnames{1} = 'erfnoopt_ValidHitObj_InvalidHitObj';	
fnames{2} = 'erf_ValidHitObj_InvalidHitObj';

fnames{3} = 'erfnoopt_ValidCue_ValidObj';	
fnames{4} = 'erf_ValidCue_ValidObj';

fnames{5} = 'erfnoopt_HCHObj_MissObj';	
fnames{6} = 'erf_HCHObj_MissObj';

suffix = {'_ValidHitObj_InvalidHitObj','_ValidCue_ValidObj','_HCHObj_MissObj'};

fh = {};
for c= 1:length(fnames)/2
    fh{c} = figure;
    
    nsub = 18;
    AUCno = [];
    TP = {}; FP = {};
    for s= 1:nsub
        %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_ValidCue_ValidObj']);
        %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_ValidHitObj_InvalidHitObj']);
        %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_HCHObj_MissObj']);
        
        load([wdir,'single_subject/erfnoopt_sub',num2str(s),suffix{c}]);
        
        [A, tp, fp] = roc(y,yhat);
        TP{s} = tp;
        FP{s} = fp;
        %plot(fp,tp,'k--','Linewidth',1);
        AUCno = [AUCno A];
    end
    
    fpi = 0:0.01:1; FPi = []; TPi = [];
    for s = 1:nsub
        fp = FP{s};
        tp = TP{s};
        %plot(fp,tp,'k--','Linewidth',1); hold on
        
        [fp, id] = unique(fp);
        tp = tp(id);
        %tpi = spline(fp,tp,fpi);
        tpi = interp1(fp,tp,fpi);
        %plot(fpi,tpi,'r--','Linewidth',1);
        %clf
        
        FPi = [FPi; fpi];
        TPi = [TPi; tpi];
    end
    P = prctile(TPi,[25 50 75],1);
    %fill([fpi'; flipdim(fpi',1)], [P(1,:)'; flipdim(P(3,:)',1)], [7 7 7]/8);%, 'EdgeColor', [7 7 7]/8);
    hold on;
    %plot(fpi,P(2,:),'k','Linewidth',2);
    plot(fpi,mean(TPi),'k','Linewidth',2);
end

%for f = 1:length(fnames)
f = 1;
for c= 1:length(fnames)/2
    figure(fh{c});

    fnames{f}
    load([wdir,fnames{f}])
    [A, tp, fp] = roc(y,yhat);
    plot(fp,tp,'b','Linewidth',2); hold on
    
    fnames{f+1}
    load([wdir,fnames{f+1}])
    [A, tp, fp] = roc(y,yhat); 
    plot(fp,tp,'r','Linewidth',2);
    
    chance=(1:size(y))/length(y);
    plot(chance,chance,'k--');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    
    f= f+2;
end
% end

