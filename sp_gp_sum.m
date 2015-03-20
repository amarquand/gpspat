function [nlml,dnlml,hyp] = sp_gp_sum(hyp, X, Y, opt)

opt.type2ml = false;

[NLML,DNLML] = sp_gp_cluster_batch(hyp, X, Y, opt);

nlml  = sum(NLML);
dnlml = rewrap(hyp,sum(DNLML,2));
end