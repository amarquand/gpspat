clear all, close all, seed = 4; randn('seed',seed); rand('seed',seed)
dev = @(x,y) max(abs(x(:)-y(:)))/max([max(abs(x(:))),max(abs(y(:))),1]);

addpath infGrid

cs = 1;                                              % toggle between test cases
if cs==1
  nx = [9,11,5]; Dx = [1,2,3]; xg = cell(numel(nx),1);    % grid data generation
  for i=1:numel(nx), xg{i} = randn(nx(i),Dx(i)); end
  x = covGrid('expand',xg);                                          % full data
  cov = {{@covSEard},{@covRQard},{@covSEiso}};
  hyp.cov = rand(eval(covGrid(cov,xg)),1);
  y = x(:,1).^2 + x(:,2) - log(abs(x(:,3)));
elseif cs==2
  nx = [37,35]; Dx = [1,1]; xg = {linspace(0,2,nx(1))',linspace(0,2,nx(2))'};
  [x2,x1] = meshgrid(xg{2},xg{1});
  x = [x1(:),x2(:)]; clear x1 x2
  cov = {{@covSEiso},{@covSEiso}};
  hyp.cov = zeros(4,1);
  y = x(:,1).^2 + x(:,2);
else
  nx = 950; Dx = 1;
  xg = {linspace(0,2,nx)'};
  x = xg{1}; x = x+0.2*(x(2)-x(1));
  cov = {@covSEiso};
  hyp.cov = zeros(2,1);
  y = x.^2 + x;  
end

N = prod(nx); idx = (1:2:N)';
mean = {@meanConst}; hyp.mean = 0.2; y = y(idx);
z = randn(56,sum(Dx));
sn = 0.1;  hyp.lik = log(sn);

cv = cell(size(cov));                   % full covariance via pointwise products
for i=1:numel(cv), cv{i} = {@covMask,{sum(Dx(1:i-1))+(1:Dx(i)),cov{i}}}; end
cv = {@covProd,cv}; j = 1;

K  = feval(cv{:}, hyp.cov, x);
Kg = covGrid(cov, xg, hyp.cov, idx);           % full covariance matrix via kron

if N<500                                            % verify auxiliary functions
  dK   = feval(cv{:}, hyp.cov, x, [],     j);
  dgK  = feval(cv{:}, hyp.cov, x(idx,:), 'diag');
  ddgK = feval(cv{:}, hyp.cov, x(idx,:), 'diag', j);
  Kz   = feval(cv{:}, hyp.cov, x(idx,:), z);
  dKz  = feval(cv{:}, hyp.cov, x(idx,:), z,      j);
  dKg  = covGrid(cov, xg, hyp.cov, idx, [], j);
  for i=1:numel(Kg)                                            % expand Toeplitz
    if iscell(Kg{i}) && strcmp(Kg{i}{1},'toep')
      Kg{i} = toeplitz(Kg{i}{2}); dKg{i} = toeplitz(dKg{i}{2});
    end
  end
  dgKg  = covGrid(cov, xg, hyp.cov, idx, 'diag');
  ddgKg = covGrid(cov, xg, hyp.cov, idx, 'diag', j);
  Kzg   = covGrid(cov, xg, hyp.cov, idx, z);
  dKzg  = covGrid(cov, xg, hyp.cov, idx, z,      j);
  K1 = 1; for i=1:numel(Kg), K1 = kron(Kg{i},K1); end

  err = dev(dgK,dgKg) + dev(K,kronmvm(Kg,eye(prod(nx))))+dev(K,K1)+ dev(Kz,Kzg);
  fprintf('\nverify matrix construction %1.3e\n',err)
  err = dev(ddgK,ddgKg) + dev(dK,kronmvm(dKg,eye(prod(nx)))) + dev(dKz,dKzg);
  fprintf('verify derivatives %1.3e\n',err)
  As = {randn(4,3),randn(5,2),randn(7,4)};
  A = kron(kron(As{3},As{2}),As{1});
  b = randn(2*3*4,9); c = A*b;
  err = max(max(abs(c-kronmvm(As,b))))+max(max(abs(A'*c-kronmvm(As,c,1))));
  fprintf('verify kronmvm %1.2e\n',err)
  B = randn(20,100); B = B'*B;
  [V,E] = eigr(B);
  err = norm(V'*V-eye(size(E,1))) + norm(V*E*V'-B);
  fprintf('verify eigr %1.3e\n\n',err)
end

idz = (1:3:N)';
tic
  [nlZ dnlZ post]   = gp(hyp, [], mean, cv, [], x(idx,:), y);
  [ymu ys2 fmu fs2] = gp(hyp, [], mean, cv, [], x(idx,:), y, x(idz,:));
fprintf('dense inf/pred took %3.1fs\n',toc)
covg = {@covGrid,cov,xg};
tic
  opt.cg_maxit = 500; opt.cg_tol = 1e-4;
  [postg nlZg dnlZg] = infGrid(hyp, mean, covg, 'likGauss', idx, y, opt);
  [ymug ys2g fmug fs2g] = gp(hyp, @infGrid, mean, covg, [], idx, postg, idz);
fprintf('grid inf/pred took %3.1fs\n',toc)

err = [dev(postg.alpha,post.alpha),dev(postg.sW,post.sW)];
fprintf('verify infGrid inference  %1.3e\n',sum(err))
fprintf('verify infGrid nlZ        %1.3e\n',dev(nlZ,nlZg))
fprintf('verify infGrid dnlZ       %1.3e\n',dev(unwrap(dnlZ),unwrap(dnlZg)))
err = [dev(ymu,ymug), dev(ys2,ys2g), dev(fmu,fmug), dev(fs2,fs2g)];
fprintf('verify infGrid prediction %1.3e\n',sum(err))

if N<500
  % compare iterative solution of linear systems
  m = feval(mean{:}, hyp.mean, x(idx,:));
  b = y-m;
  Ki = K(idx,idx)+sn^2*eye(numel(idx));
  cut = @(x, idx) x(idx);
  ins = @(xi,idx) accumarray(idx,xi,[prod(nx),1]);
  mvm = @(x)      kronmvm(Kg,x)+sn^2*x;
  afn = @(xi)     cut(mvm(ins(xi,idx)),idx);
  tol = 1e-10;    maxit = min(650,prod(nx));
  arg = '(afn,b,tol,maxit)';
  for op = {'mldiv ','minres','pcg   ','conjgrad'}
    tic
    if strcmp(op{:},'mldiv '), al = Ki\b; i = size(Ki,1);
    else [txt,al,f,r,i] = evalc([op{:},arg]); end
    t = toc;
    fprintf([op{:},') res=%1.3e, i=%d, t=%1.2fs\n'],norm(b-Ki*al)/norm(b),i,t)
  end
end

rmpath infGrid