function Kf = v2m(lf,T)

Lf        = zeros(T);
id        = tril(true(T));
Lf(id)    = lf;
Kf        = Lf*Lf';