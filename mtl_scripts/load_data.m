function [X,Y,ID,mask,classes] = load_data(root)

nsub  = 18;
nrun  = 10;
debug = false;

try 
    root;
catch
    root = '/cns_zfs/mlearn/public_datasets/openfmri/posner/';
end

mask = [root,'masks/SPM_mask_46x55x39.img'];
Nm   = nifti(mask);
mask = Nm.dat(:,:,:);
mask = find(mask(:));

classes = {'ValidHCHitCue',   'ValidHCHitObj',   'ValidLCHitCue',...
           'ValidLCHitObj',   'ValidMissCue',    'ValidMissObj',...
           'InvalidHCHitCue', 'InvalidHCHitObj', 'InvalidLCHitCue',...
           'InvalidLCHitObj', 'InvalidMissCue',  'InvalidMissObj'};

X  = zeros(nsub*nrun*length(classes),length(mask));
Y  = zeros(nsub*nrun*length(classes),length(classes));
ID = zeros(nsub*nrun*length(classes),3);
ct = 1;
for s = 1:nsub
    fprintf('> Subject %d ... \n',s);
    for r = 1:nrun
        if debug,fprintf('>> Run %d ... \n',r); end
        sdir = [root,'sub',num2str(s,'%03.0f'),'/BOLD/task001_run',num2str(r,'%03.0f'),'/'];
        
        for c = 1:length(classes);
            N = nifti([sdir,'beta_',num2str(c,'%04.0f'),'.img']); 
            
            if debug,fprintf('Subject %d, run %d, beta %d: ',s,r,c); end
            
            found = false;
            for c2 = 1:length(classes)
                if ~isempty(regexp(N.descrip,classes{c2},'match'))
                    if debug, fprintf('matched to class %d.\n',c2); end
                                   
                    vol      = N.dat(:,:,:);
                    vol      = vol(:);
                    vol      = vol(mask);
                    X(ct,:)  = vol'; 
                    Y(ct,c2) = 1;            
                    
                    ID(ct,1) = s;
                    ID(ct,2) = r;
                    ID(ct,3) = c;
                    ct = ct+1;
                    found = true;
                end
            end
            if ~found && debug
                fprintf(' NOT FOUND.\n');
            end
        end
    end
    fprintf('> done.\n');        
end
% trim data matrices (to accommodate subjects not having all conditions)
X  = X(1:ct,:);
Y  = Y(1:ct,:);
ID = ID(1:ct,:);

% remove nans
X(isnan(X)) = 0;

% the following method removes too many voxels
%mask2 = sum(isnan(X)) > 0;
%X = X(:,mask2);
