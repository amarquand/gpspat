wdir = '/cns_zfs/mlearn/public_datasets/openfmri/posner/gp_da/';

colours ={,'m','g','r','b','k'};
n_per_gp = 5;

leg_entries = {'OS pooled','OS-MTL','EP pooled','EP-MTL (F)','EP-MTL (R)','OS single sub','EP single sub'};
c = 1;
fnames{c} = 'noopt_ValidHitObj_InvalidHitObj';     c = c+1;
fnames{c} = 't2ml_ValidHitObj_InvalidHitObj';     c = c+1;
fnames{c} = 'erfnoopt_ValidHitObj_InvalidHitObj'; c = c+1;	
fnames{c} = 'erf_ValidHitObj_InvalidHitObj';      c = c+1;
fnames{c} = 'erfmt_ValidHitObj_InvalidHitObj';      c = c+1;

fnames{c} = 'noopt_ValidCue_ValidObj';    c = c+1;	
fnames{c} = 't2ml_ValidCue_ValidObj';     c = c+1;
fnames{c} = 'erfnoopt_ValidCue_ValidObj'; c = c+1;	
fnames{c} = 'erf_ValidCue_ValidObj';      c = c+1;
fnames{c} = 'erfmt_ValidCue_ValidObj';      c = c+1;

fnames{c} = 'noopt_HCHObj_MissObj';    c = c+1;
fnames{c} = 't2ml_HCHObj_MissObj';     c = c+1;
fnames{c} = 'erfnoopt_HCHObj_MissObj'; c = c+1;
fnames{c} = 'erf_HCHObj_MissObj';      c = c+1;
fnames{c} = 'erfmt_HCHObj_MissObj';      c = c+1;

suffix = {'_ValidHitObj_InvalidHitObj','_ValidCue_ValidObj','_HCHObj_MissObj'};


fh = {};
for c= 1:length(fnames)/n_per_gp
    fh{c} = figure;
end

f = 1;
for c= 1:length(fnames)/n_per_gp
    figure(fh{c})
    
    for d = 1:n_per_gp
    fnames{f+d-1}
    load([wdir,fnames{f+d-1}])    
    [A, tp, fp] = roc(y,yhat);
    A
    plot(fp,tp,colours{d},'Linewidth',2); hold on
    disp '----'
    end
    
    f= f+d;
end

prefix = {'erfnoopt','noopt'}; colours2 = {[0.5 0.7 0.5],[105 105 105]/255};
for p = 1:length(prefix)
    prefix{p}
    for c= 1:length(fnames)/n_per_gp
        figure(fh{c});
        
        nsub = 18;
        AUCno = [];
        TP = {}; FP = {};
        for s= 1:nsub
            %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_ValidCue_ValidObj']);
            %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_ValidHitObj_InvalidHitObj']);
            %load([wdir,'single_subject/erfnoopt_sub',num2str(s),'_HCHObj_MissObj']);
            
            %load([wdir,'single_subject/erfnoopt_sub',num2str(s),suffix{c}]);
            load([wdir,'single_subject/',prefix{p},'_sub',num2str(s),suffix{c}]);
            
            [A, tp, fp] = roc(y,yhat);
            TP{s} = tp;
            FP{s} = fp;
            %plot(fp,tp,'k--','Linewidth',1);
            AUCno = [AUCno A];
        end
        AUC = mean(AUCno)
        
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
        plot(fpi,mean(TPi),'Color',colours2{p},'Linewidth',2);
        
    end
end

f = 1;
for c= 1:length(fnames)/n_per_gp
    figure(fh{c})
    legend(leg_entries,'Location','SouthEast')
    for d = 1:n_per_gp
        %fnames{f}
        load([wdir,fnames{f+d-1}])
        [A, tp, fp] = roc(y,yhat);
        plot(fp,tp,colours{d},'Linewidth',2); hold on
    end
    
    chance=(1:size(y))/length(y);
    plot(chance,chance,'k--');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    
    f= f+d;
end



