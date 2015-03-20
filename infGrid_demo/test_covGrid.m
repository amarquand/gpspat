clear all, close all, seed = 4; randn('seed',seed); rand('seed',seed)
dev = @(x,y) max(abs(x(:)-y(:)))/max([max(abs(x(:))),max(abs(y(:))),1]);

if 1                                             % grid conversion functionality
  nx = [15,4,16,5]; Dx = [1,1,1,1]; xg = cell(numel(nx),1);
  for i=1:numel(nx), xg{i} = randn(nx(i),Dx(i)); end
  [z,nz,Dz]  = covGrid('expand',xg);
  [a,b,c,d] = ndgrid(xg{:}); x = [a(:),b(:),c(:),d(:)];
  err = dev(x,z)+dev(nx,nz)+dev(Dx,Dz);
  fprintf('err wrt ndgrid (p=%d) %1.2e\n',numel(nx),err)
  if err>1e-10, error('Problem here.'), end
  zg = covGrid('factor',{z,nz,Dz});
  err = 0; for i=1:numel(nx), err = err+dev(zg{i},xg{i}); end
  fprintf('self consist   (p=%d) %1.2e\n',numel(nx),err)
  if err>1e-10, error('Problem here.'), end

  nx = [2,4]; Dx = [1,1]; xg = cell(numel(nx),1);
  for i=1:numel(nx), xg{i} = randn(nx(i),Dx(i)); end
  [z,nz,Dz]  = covGrid('expand',xg);
  [a,b] = meshgrid(xg{:}); vec = @(x) x(:); x = [vec(a'),vec(b')];
  err = dev(x,z)+dev(nx,nz)+dev(Dx,Dz);
  fprintf('err wrt ndgrid (p=%d) %1.2e\n',numel(nx),err)
  if err>1e-10, error('Problem here.'), end
  zg = covGrid('factor',{z,nz,Dz});
  err = 0; for i=1:numel(nx), err = err+dev(zg{i},xg{i}); end
  fprintf('self consist   (p=%d) %1.2e\n',numel(nx),err)
  if err>1e-10, error('Problem here.'), end

  nx = [6,9,11,5,2]; Dx = [3,2,1,4,2]; xg = cell(numel(nx),1);
  for i=1:numel(nx), xg{i} = randn(nx(i),Dx(i)); end
  [z,nz,Dz]  = covGrid('expand',xg);
  zg  = covGrid('factor',{z,nz,Dz});
  err = 0; for i=1:numel(nx), err = err+dev(zg{i},xg{i}); end
  err = err+dev(nx,nz)+dev(Dx,Dz);
  fprintf('self consist   (p=%d) %1.2e\n',numel(nx),err)
  if err>1e-10, error('Problem here.'), end
end
