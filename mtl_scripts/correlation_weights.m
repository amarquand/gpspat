root = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';

names{1}  = [root,'gp_da/single_subject/weights/erfnoopt_ValidHitObj_InvalidHitObj/erfnoopt_ValidHitObj_InvalidHitObj'];
names{2}  = [root,'gp_da/weights/erfnoopt_ValidHitObj_InvalidHitObj/erfnoopt_ValidHitObj_InvalidHitObj'];
names{3}  = [root,'gp_da/weights/erf_ValidHitObj_InvalidHitObj/erf_ValidHitObj_InvalidHitObj'];
names{4}  = [root,'gp_da/weights/erfmt_ValidHitObj_InvalidHitObj/erfmt_ValidHitObj_InvalidHitObj'];

titles = {'EP single subject', 'EP pooled', 'EP-MTL (F)', 'EP-MTL (R)'}

mname = [root,'masks/SPM_mask_46x55x39.img'];
Nm = nifti(mname);
dm = Nm.dat.dim;
mask = reshape(Nm.dat(:,:,:),prod(dm(1:3)),1);
mask(isnan(mask)) = 0;
mask = logical(mask);

Nfold = 10;
Nsub  = 18;

Wa = {}; 
for n = 1:length(names)
    name = names{n}
    W = [];
    for s = 1:Nsub
        
        fprintf('Loading subject %d\n',s)
        for f = 1:Nfold
            try
                N = nifti([name,'_W_fold_',num2str(f,'%02.0f'),'_task',num2str(s),'.img']);
                w = N.dat(:,:,:);
                w = w(:);% ./ norm(w(:));
                w = w(mask);
                %w = w ./ max(abs(w));
                
                %             N = nifti([name,'_XW_fold_',num2str(f,'%02.0f'),'_task',num2str(s),'.img']);
                %             w = [];
                %             for i = 1:N.dat.dim(4)
                %                 wi = N.dat(:,:,:,i);
                %                 wi = wi(:);
                %                 wi = wi(mask);
                %                 w = [w wi];
                %             end
                W = [W; w'];
            catch
                fprintf('couldn''t read subject %d fold %d\n',s,f)
            end
        end
    end
    Wa{n} = W;
end
figure
Wam = {}; Rho = {};
for n = 1:length(names)
    id = 1:10;
    
    Ws = []; rho = [];
    for s = 1:Nsub
        Ws = [Ws; mean(Wa{n}(id,:))];
        rho = [rho, mean(mean(corr(Wa{n}(id,:)')))];
        if s == 8
            id = id + 6;           
        else
            id = id + 10;
        end
    end
    Wam{n} =Ws;
    Rho{n} = rho;
    subplot(2,2,n)
    %imagesc(corr(Wam{n}'), [-1 1]);
    hintonDiagram(corr(Wam{n}'));
    title(titles{n})
    xlabel 'task (subject)'
    if n < 3, box on; end
    ylabel 'task (subject)'
end

figure
for n = 1:length(names)
    subplot(2,2,n)
    imagesc(corr(Wa{n}'), [-1 1]);
    %hintonDiagram(corr(Wa{n}'));
end


