% kronmvm, part of infGrid

function b = kronmvm(As,x,transp)
if nargin>2 && ~isempty(transp) && transp   % transposition by transposing parts
  for i=1:numel(As), As{i} = As{i}'; end
end
m = zeros(numel(As),1); n = zeros(numel(As),1);                  % extract sizes
for i=1:numel(n)
  if iscell(As{i}) && strcmp(As{i}{1},'toep')
    m(i) = size(As{i}{2},1); n(i) = size(As{i}{2},1);
  else [m(i),n(i)] = size(As{i});
  end
end
d = size(x,2);
b = x;
for i=1:numel(n)
  a = reshape(b,[prod(m(1:i-1)), n(i), prod(n(i+1:end))*d]);    % prepare  input
  tmp = reshape(permute(a,[1,3,2]),[],n(i))*As{i}';
  b = permute(reshape(tmp,[size(a,1),size(a,3),m(i)]),[1,3,2]);
end
b = reshape(b,prod(m),d);                        % bring result in correct shape