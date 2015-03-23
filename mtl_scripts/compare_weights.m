root = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';

names{1}  = [root,'gp_da/weights/t2ml_HCHObj_MissObj/t2ml_HCHObj_MissObj'];
names{2}  = [root,'gp_da/weights/noopt_HCHObj_MissObj/noopt_HCHObj_MissObj'];

%names{1}  = [root,'gp_da/weights/erf_ValidHitObj_InvalidHitObj/erf_ValidHitObj_InvalidHitObj'];
%names{2}  = [root,'gp_da/weights/erfnoopt_ValidHitObj_InvalidHitObj/erfnoopt_ValidHitObj_InvalidHitObj'];

names{1}  = [root,'gp_da/weights/erf_HCHObj_MissObj/erf_HCHObj_MissObj'];
names{2}  = [root,'/gp_da/weights/erfnoopt_HCHObj_MissObj/erfnoopt_HCHObj_MissObj'];
names{3}  = [root,'/gp_da/single_subject/weights/erfnoopt_HCHObj_MissObj/erfnoopt_HCHObj_MissObj'];

mname = [root,'masks/SPM_mask_46x55x39.img'];
Nm = nifti(mname);
dm = Nm.dat.dim;
mask = reshape(Nm.dat(:,:,:),prod(dm(1:3)),1);
mask(isnan(mask)) = 0;
mask = logical(mask);

Nfold = 10;
Nsub  = 18;

Wa = {}; %Wam = {};
for c = 1:length(names)
    name = names{c};
    W = [];
    for s = 1:Nsub
        
        fprintf('Loading subject %d\n',s)
        for f = 1:Nfold
            %if f ~= 8
                try
                    N = nifti([name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(s)]);
                    w = N.dat(:,:,:);
                    %w = w(:) ./ norm(w(:));
                    w = w(mask);
                    W = [W; w'];
                catch
                    fprintf('couldn''t read subject %d fold %d\n',s,f)
                end
            %end
        end
    end
    Wa{c} = W;
    
    if c < 3
    Wm = [];
    for f = 1:Nfold
        N = nifti([name,'_W_fold_',num2str(f,'%02.0f'),'_meantask']);
        wm = N.dat(:,:,:);
        wm = wm(mask);
        Wm = [Wm; wm'];
    end
    Wam{c} = Wm;
    end
end

rc = {};
for c = 1:2
    c
    subplot(1,2,c)
    Wc = Wa{c};
    %Wc = (Wa{c} - repmat(mean(Wa{c}),size(Wa{c},1),1)) ./ repmat(std(Wa{c}),size(Wa{c},1),1);
    R = corr(Wc');
    imagesc(R)
    block = 1:10;
    r =[];
    for b = 1:Nsub
        b
        range = (b-1)*max(block)+block;
        Rb = R(range,range);
        r = [r; mean(mean(Rb))];
    end
    rc{c} = r;
end
