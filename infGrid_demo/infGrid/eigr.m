% eigr, part of infGrid

% Real eigenvalues and eigenvectors up to the rank of a real symmetric matrix.
% Decompose A into V*D*V' with orthonormal matrix V and diagonal matrix D.
% Diagonal entries of D obave the rank r of the matrix A as returned by
% the call rank(A,tol) are zero.
function [V,D] = eigr(A,tol)
[V,D] = eig((A+A')/2); n = size(A,1);    % decomposition of strictly symmetric A
d = max(real(diag(D)),0); [d,ord] = sort(d,'descend');        % tidy up and sort
if nargin<2, tol = size(A,1)*eps(max(d)); end, r = sum(d>tol);     % get rank(A)
d(r+1:n) = 0; D = diag(d);                % set junk eigenvalues to strict zeros
V(:,1:r) = real(V(:,ord(1:r))); V(:,r+1:n) = null(V(:,1:r)'); % ortho completion