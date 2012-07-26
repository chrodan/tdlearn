  function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp, ...
                             dMdw, dSdw, dCdw, dMdh, dSdh, dCdh] = congp(par, m, s)

% Gaussian process policy. Compute mean, variance and input output covariance of
% the action using a Gaussian process policy function, when the input is
% Gaussian. The GP is parameterized using a pseudo training set of
% N cases. Optionally, compute partial derivatives wrt the input parameters.
%
% inputs:
% par      policy parameters (struct)
% par.w    policy pseudo targets, N by D
% par.p    policy pseudo inputs, N by d
% par.hyp  GP log hyperparameters, (d+2)*D by 1
% m        mean of state distribution, d by 1
% s        covariance matrix of state distribution, d by d
%
% outputs:
% M        mean of the action, D by 1
% S        variance of action, D by D
% C        covariance input and action, d by D
% dMdm     derivative of mean action wrt mean of state, D by d
% dSdm     derivative of variance of action wrt mean of state, D by D by d
% dCdm     derivative of covariance wrt mean of state, d by D by d
% dMds     derivative of mean action wrt variance, D by d by d
% dSds     derivative of action variance wrt variance, D by D by d by d
% dCds     derivative of covariance wrt variance, d by D by d by d
% dMdp     derivative of mean action wrt GP pseudo inputs, d by N by d
% dSdp     derivative of action variance wrt GP pseudo inputs, D by D by N by d
% dCdp     derivative of covariance wrt GP pseudo inputs, d by D by N by d
% dMdw     D by N by D
% dSdw     D by D by N by D
% dCdw     d by D by N by D
% dMdh     D by (d+2)*D
% dSdh     D by D by (d+2)*D
% dCdh     d by D by (d+2)*D
%
% Copyright (C) 2008-2009 by Carl Edward Rasmussen & Marc Deisenroth, 2009-10-26

if nargout < 4                                  % if no derivatives are required
  [M, S, C] = gpP0d(par.hyp, par.p, par.w, m, s);
else                                            % else compute derivatives, too
  [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp, dMdw, ...
             dSdw, dCdw, dMdh, dSdh, dCdh] = gpP2d(par.hyp, par.p, par.w, m, s);
end