% Function to initialise SM kernel hyperparameters

% If varargin{1} is specified, it is the number of optimisation iterations
% to run for each random restart.  Otherwise, we will just have
% initialisations and no optimisation.

function hyp = spectral_init(inf_method,hyp,meanf,lik,cov,covg,x,y,idx,nrestarts,varargin)
  
% start with user inputed hypers
  hyp_best = hyp;
  
  try
    % compute nlml for user specified hyp
    bestlik = gp(hyp, inf_method, meanf, covg, lik, idx, y);
  catch
    disp('Error with user specified hypers.');
    disp('Attempting to proceed with automatic initialisation.');
    bestlik = Inf;
  end
  
  disp(sprintf('Initialisation nlml 0: %.02f',bestlik));
  
  % try 'nrestarts' number of initialisations
  for ri=1:nrestarts
    hyp.cov = [];     % shouldn't overwrite yet
    for i=1:numel(cov)   
        % call the initialisation script for two 1D spectral mixture
        % kernels
        hyp.cov = [hyp.cov; hypinit1D('covSMfast', cov{i}{2}, x ,y)]; % want the true inputs x here
    end
    hyp.cov = log(hyp.cov);
    
    % if desired, try iter_run optimization iterations for each initialisation
    if(~isempty(varargin))
          iter_run = varargin{1};
          hyp = minimize(hyp,@gp,-iter_run,inf_method,meanf,covg,lik,idx,y);
    end
    
    % see if nlml of new initialisation is better
    try
        l = gp(hyp, inf_method, meanf, covg, lik, idx, y);
        disp(sprintf('Initialisation nlml %d: %.02f', ri, l))
        if l < bestlik
            bestlik = l;
            hyp_best = hyp;
        end
    catch
        disp('Error trying initialisation');
    end
  end
  
  hyp = hyp_best;

% initialise a 1D spectral mixture kernel
function [hypout] = hypinit1D(covtype,Q,x,y)
  hypout = [];
  switch(covtype)
    case 'covSMfast'
        
        % NOTE TO USER: SET FS= 1/[MIN GRID SPACING] FOR YOUR APPLICATION
        % Fs is the sampling rate
        Fs = 1;   % 1/[grid spacing].  
             
        % Deterministic weights (fraction of variance)
        % Set so that k(0,0) is close to the empirical variance of the data.
        
        wm = std(y);
        w0 = wm/sqrt(Q)*ones(Q,1);
       
        w0 = w0.^2; % parametrization for covSMfast
        
        hypout = [w0];        
        
        % Uniform random frequencies
        % Fs/2 will typically be the Nyquist frequency
        mu = max(Fs/2*rand(Q,1),1e-8);

        hypout = [hypout; mu];
        
        % Truncated Gaussian for length-scales (1/Sigma)
        sigmean = length(unique(x))*sqrt(2*pi)/2;
        hypout = [hypout; 1./(abs(sigmean*randn(Q,1)))];
  end
  % Todo: Add SE and MA kernels.
  

  
