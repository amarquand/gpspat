function [NLML,DNLML,HYP] = sp_gp_cluster_batch(hyp, X, Y, opt)

ones(10)*ones(10); % stupid hack to get matlab to work properly

try opt.type2ml;    catch, opt.type2ml = false; end
try opt.usecluster; catch, opt.usecluster = true; end
try opt.walltime;   catch, opt.walltime = 300; end
try opt.memory;     catch, opt.memory = 100*1024^2; end
try opt.debug;      catch, opt.debug = false; end

D   = size(X,2);
T   = size(Y,2);  % number of tasks

Njobs = T;
Nperjob = 1;

NLML  = zeros(T,1);
DNLML = zeros(length(unwrap(hyp)),T);
HYP   = zeros(T,length(unwrap(hyp)));

Xb   = cell(Njobs,1);  yb  = cell(Njobs,1);
optb  = cell(Njobs,1); hypb  = cell(Njobs,1);
for b = 1:Njobs
    if b < Njobs
        id = (1:Nperjob)+(b-1)*Nperjob;
    else
        id = ((b-1)*Nperjob+1):size(Y,2);
    end
    
    yb{b}   = Y(:,id);
    optb{b} = opt;
    if size(X,3) > 1
        Xb{b} = X(:,:,b);
    else
        Xb{b} = X;
    end
    hypb{b}  = hyp;
end

if opt.usecluster
    % use batch queue
    cfun.fname = 'sp_gp_cluster_job';
    cfun.fdeps = {};
    cfun.executable = '/home/mrstats/andmar/sfw/gpspat/sp_gp_cluster_job.sh';
    [nlmls,dnlmls,hyps] = qsubcellfun(cfun,hypb,Xb,yb,optb,'memreq', opt.memory, 'timreq', opt.walltime,'queue','batch');
    
    % use matlab queue
    %[nlmls,dnlmls,hyps] = qsubcellfun('sp_gp_cluster_job',hypb,Xb,yb,optb,'memreq', opt.memory, 'timreq', opt.walltime); 
    
    qsub_cleanup;
else % run sequentially
    nlmls = cell(Njobs,1); dnlmls = cell(Njobs,1); hyps = cell(Njobs,1); 
    for n = 1:Njobs
        if opt.debug, fprintf('running job %d of %d...\n',n,Njobs'); end
        [nlmls{n},dnlmls{n},hyps{n}] = sp_gp_cluster_job(hypb{n},Xb{n},yb{n},optb{n});
    end
end

for b = 1:Njobs
    if b < Njobs
        id = (1:Nperjob)+(b-1)*Nperjob;
    else
        id = ((b-1)*Nperjob+1):size(Y,2);
    end
    NLML(id)    = nlmls{b};
    DNLML(:,id) = dnlmls{b};    
    HYP(id,:) = hyps{b};
end
end


function fstem = qsub_cleanup

user     = getenv('USER');
pid      = feature('getpid');
[~,host] = system('hostname -s');
host     = strtrim(host);
host     = regexprep(host,'-','_');
fstem    = [user,'_',host,'_p',num2str(pid)];


fprintf('Cleaning up... ');
system(['rm -f ',fstem,'*']); 
fprintf('done.\n');
end
