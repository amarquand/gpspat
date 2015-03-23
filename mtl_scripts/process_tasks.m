function [X,y,ID,Y,Ys,Yc] = process_tasks(Xa,Ya,ID)

% define some variables to improve readability
subjects  = unique(ID(:,1));
fmriTasks = unique(ID(:,3));

disp('Checking for missing scans ...');
bad = false(size(ID,1),1);
for s = 1:length(subjects)
    sid = ID(:,1) == s;
    for r = 1:length(unique(ID(sid,2)))
        srid = sid & ID(:,2) == r;
        
        if numel(unique(ID(srid,3))) < size(Ya,2)
            fprintf('> Excluding subject %d, run %d (missing data)\n',subjects(s),r)
            bad(srid) = true;
        end
    end
end
disp('Error checking done.');
ID = ID(~bad,:);
Ya = Ya(~bad,:);
X  = Xa(~bad,:);

% Ya contains one column per fmri task. Now we reparametrise so that Y is
% a cell array containing one cell per subject per fmri task. We also
% duplicate the data matrix to accommodate the binary classification.
%
% NOTE: at this stage the noise parameters for each class within each
%       subject are NOT coupled!
%
T   = length(subjects);
N   = length(ID(:,1));
Y   = zeros(N,T);   % main regression matrix
Yc  = cell(1,T);    % cell array of individual tasks
Ys  = zeros(N,T);   % indicator matrix for subjects
ct  = 1; 
for s = 1:length(subjects)
    sid = ID(:,1) == subjects(s);
    Yc{ct} = Ya(sid,1)';
    Ys(sid,ct) = 1;
    
    tid = sid & ID(:,3) == fmriTasks(1);
    Y(tid,ct) = 1;
    
    ID(sid,4) = ct;
    ct = ct + 1;
end

% Optimal scoring
classifiers = unique(ID(:,1));
Nclassifiers = length(classifiers);
for c = 1:Nclassifiers
    Yos   = opt_score([Yc{c}', 1-Yc{c}']);
    Yc{c} = Yos(:,1)';
end
% % Optimal scoring
% Nclassifiers = max(ID(:,1));
% for c = 1:Nclassifiers
%     Yos   = opt_score([Yc{c}', 1-Yc{c}']);
%     Yc{c} = Yos(:,1)';
% end

% generate final labels
y = [Yc{:}]';