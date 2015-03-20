function [Xidx, opt, Xidx2] = sp_configure_grid(X0, opt0, tr, spdata)

% note: tr indexes spatial coordinates (in X0)!!!

D = size(X0,2);

opt = opt0;
try tr; catch, tr = true(size(X0,1),1); end

% set up grid
xg = {(min(X0(:,1)):2:max(X0(:,1)))',...
      (min(X0(:,2)):2:max(X0(:,2)))',...
      (min(X0(:,3)):2:max(X0(:,3)))'};
% % alternative method (isotropic grid)
% xmin = min(X0(:));
% xmax = max(X0(:));
% xg   = {(xmin:2:xmax)',(xmin:2:xmax)',(xmin:2:xmax)'};
cf = {opt.cov, opt.cov, opt.cov};

[xe,ng] = covGrid('expand',xg);

% find the indices for the training data
Xidx = zeros(sum(tr),1);
tridx  = 1:size(Xidx,1);
for i = 1:sum(tr)
    xi = X0(tridx(i),1) == xe(:,1);
    yi = X0(tridx(i),2) == xe(:,2);
    zi = X0(tridx(i),3) == xe(:,3);
    Xidx(i) = find(xi & yi & zi);
end

% Alternative method. This does not work because txform needs to be applied
% % trim the mask
% x0 = squeeze(sum(sum(spdata.mask,2),3) == 0);
% y0 = squeeze(sum(sum(spdata.mask,1),3) == 0);
% z0 = squeeze(sum(sum(spdata.mask,1),2) == 0);
% mask = spdata.mask(~x0,~y0,~z0);
% %mask = spdata.mask(:);
% Xidx2 = find(mask(:));
% X2 = xe(Xidx2,:); X1 = xe(Xidx,:);

% % find the indices for the test data
%Xte{b} = zeros(sum(te),1);
%teidx  = find(te);
%for i = 1:sum(te)
%    xi = Xidx(teidx(i),1) == xe(:,1);
%    yi = Xidx(teidx(i),2) == xe(:,2);
%    zi = Xidx(teidx(i),3) == xe(:,3);
%    Xte{b}(i) = find(xi & yi & zi);
%end

gopt.cg_maxit = 1000; gopt.cg_tol = 1e-3;
%opt.inf       = @(varargin) infGrid(varargin{:},gopt);
opt.inf       = @sp_infGrid;
opt.cov       = {@covGrid, cf ,xg};
opt.hyp0.cov  = repmat(opt.hyp0.cov,D,1);

if iscell(opt.mean)
    if strcmpi(func2str(opt.mean{1}),'meanPoly')
        %opt.mean = {@(varargin) meanPolyGrid3(varargin{:}, opt.cov) opt.mean{2}};
        opt.mean = {@(varargin) meanPolyGrid3(varargin{:}) opt.mean{2}};
    else
        error('unsupported mean function');
    end
end

end