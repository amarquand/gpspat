function varargout = sp_infGrid(varargin)

gopt.cg_maxit = 1000; gopt.cg_tol = 1e-3;
switch nargout
    case 1
        post = infGrid(varargin{:},gopt);
        varargout = {post};
    case 2
        [post nlZ] = infGrid(varargin{:},gopt);
        varargout = {post, nlZ};
    case 3
        [post nlZ dnlZ] = infGrid(varargin{:},gopt);
        varargout = {post, nlZ, dnlZ};
    otherwise
        error('incorrect usage of infGrid');
end
        
end
