function fstem = sp_qsub_cleanup

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