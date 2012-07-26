% Compute moments of the saturating function e*(9*sin(x(i))+sin(3*x(i)))/8, 
% where x sim N(m,v) and i is a (possibly empty) set of I indices. The optional
% e scaling factor is a vector of length I. Optionally, compute derivatives of
% the moments.
%
% Copyright (C) 2012 by Carl Edward Rasmussen, Andrew McHutchon and
%                                                    Marc Deisenroth 2012-06-21

function [M, S, C, dMdm, dSdm, dCdm, dMdv, dSdv, dCdv] = gSat(m, v, i, e)

% m     mean vector of Gaussian                                     [ d       ]
% v     covariance matrix                                           [ d  x  d ]
% i     I length vector of indices of elements to augment
% e     I length optional scale vector (defaults to unity)
%
% M     output means                                                [ I       ]
% V     output covariance matrix                                    [ I  x  I ]
% C     inv(v) times input-output covariance                        [ d  x  I ]
% dMdm  derivatives of M w.r.t m                                    [ I  x  d ]
% dVdm  derivatives of V w.r.t m                                    [I*I x  d ]
% dCdm  derivatives of C w.r.t m                                    [d*I x  d ]
% dMdv  derivatives of M w.r.t v                                    [ I  x d*d]
% dVdv  derivatives of V w.r.t v                                    [I*I x d*d]
% dCdv  derivatives of C w.r.t v                                    [d*I x d*d]

d = length(m); I = length(i); i = i(:)';
if nargin < 4; e = ones(1, I); end; e = e(:)';

P = [eye(d); 3*eye(d)];                                        % augment inputs
ma = P*m;    madm = P;
va = P*v*P'; vadv = kron(P,P); va = (va+va')/2;

[M2, S2, C2, Mdma, Sdma, Cdma, Mdva, Sdva, Cdva] ...
                                            = gSin(ma, va, [i d+i], [9*e e]/8);

P = [eye(I) eye(I)]; Q = [eye(d) 3*eye(d)];
M = P*M2;                                                                % mean
S = P*S2*P'; S = (S+S')/2;                                           % variance
C = Q*C2*P';                                    % inv(v) times input-output cov

if nargout > 3                                        % derivatives if required
  dMdm = P*Mdma*madm;         dMdv = P*Mdva*vadv;
  dSdm = kron(P,P)*Sdma*madm; dSdv = kron(P,P)*Sdva*vadv;
  dCdm = kron(P,Q)*Cdma*madm; dCdv = kron(P,Q)*Cdva*vadv;
end
