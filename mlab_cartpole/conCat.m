function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, ...
                        dMdp, dSdp, dCdp] = conCat(con, sat, policy, m, s)
% conCat is a utility function which concatenates a controller "con" with
% a saturation function "sat" such as eg gSat.m and composes their
% derivatives.
% 
% Carl Edward Rasmussen and Andrew McHutchon, 2012-06-25
% Edited by Joe Hall 2012-07-03

maxU=policy.maxU; E=length(maxU); D=length(m); F=D+E; j=D+1:F; i=1:D;
M = zeros(F,1); M(i) = m; S = zeros(F); S(i,i) = s;         % init M and S

if nargout < 4                                       % without derivatives

  [M(j), S(j,j), Q] = con(policy, m, s); 
  q = S(i,i)*Q; S(i,j) = q; S(j,i) = q';
  [M, S, R] = sat(M, S, j, maxU);
  C = [eye(D) Q]*R;
  
else                                                    % with derivatives
  Mdm = zeros(F,D); Sdm = zeros(F*F,D); Mdm(1:D,1:D) = eye(D);
  Mds = zeros(F,D*D); Sds = kron(Mdm,Mdm);
  
  X = reshape(1:F*F,[F F]); XT = X';                  % vectorised indices
  I=0*X;I(j,j)=1;jj=X(I==1)'; I=0*X;I(i,j)=1;ij=X(I==1)'; ji=XT(I==1)';
  
  % Controller -----------------------------------------------------------
  [M(j), S(j,j), Q, Mdm(j,:), Sdm(jj,:), dQdm, Mds(j,:), ...
                  Sds(jj,:), dQds, Mdp, Sdp, dQdp] = con(policy, m, s);
  q = S(i,i)*Q; S(i,j) = q; S(j,i) = q';
  
  SS = kron(eye(E),S(i,i)); QQ = kron(Q',eye(D));
  Sdm(ij,:) = SS*dQdm;      Sdm(ji,:) = Sdm(ij,:);
  Sds(ij,:) = SS*dQds + QQ; Sds(ji,:) = Sds(ij,:);

  % Saturation -----------------------------------------------------------
  [M, S, R, MdM, SdM, RdM, MdS, SdS, RdS] = sat(M, S, j, maxU);
  
  dMdm = MdM*Mdm + MdS*Sdm; dMds = MdM*Mds + MdS*Sds;
  dSdm = SdM*Mdm + SdS*Sdm; dSds = SdM*Mds + SdS*Sds;
  dRdm = RdM*Mdm + RdS*Sdm; dRds = RdM*Mds + RdS*Sds;
  
  dMdp = MdM(:,j)*Mdp + MdS(:,jj)*Sdp;
  dSdp = SdM(:,j)*Mdp + SdS(:,jj)*Sdp;
  dRdp = RdM(:,j)*Mdp + RdS(:,jj)*Sdp;

  C = [eye(D) Q]*R;
  RR = kron(R(j,:)',eye(D)); QQ = kron(eye(E),[eye(D) Q]);
  dCdm = QQ*dRdm + RR*dQdm;
  dCds = QQ*dRds + RR*dQds;
  dCdp = QQ*dRdp + RR*dQdp;
end