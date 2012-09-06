function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = conGaussd(policy, m, s)
% Compute joint predictions for multiple Gaussian Radial Basis Functions (RBFs)
% with uncertain inputs.
%
% inputs:
% policy        policy struct
%   .p          parameters to optimise
%     .ll       D by E matrix of log lengthscales
%     .cen      n by D matrix of input locations for basis functions
%     .w        n by E matrix of basis functions weights
% m             D by 1 mean of the test distribution              
% s             D by D covariance matrix of the test distribution
%
% outputs:
% M       mean of pred. distribution                            [E    ] 
% S       covariance of the pred. distribution                  [E x E]
% C       inv(s) times covariance between input and prediction  [D x E]
% dMdm    output mean   by input mean partial derivatives       [E   x D]
% dSdm    output cov    by input mean derivatives               [E*E x D]
% dCdm    inv(s)*io-cov by input mean derivatives               [D*E x D]
% dMds    ouput mean    by input covariance derivatives         [E   x D*D]
% dSds    output cov    by input covariance derivatives         [E*E x D*D] 
% dCds    inv(s)*io-cov by input covariance derivatives         [D*E x D*D]
%
% dMdp    output mean   by policy parameters                    [E     x Np] 
% dSdp    output cov    by policy parameters                    [E*E x Np] 
% dCdp    inv(s)*io-cov by policy parameters                    [D*E x Np]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
% Joe Hall, and Andrew McHutchon. 2012-07-23

cen = policy.p.cen;  w = policy.p.w; ll = policy.p.ll;

[n, D] = size(cen);         % number of examples and dimension of input space
E = size(w,2);               % number of examples and number of outputs

% initializations
M = zeros(E,1); S = zeros(E); C = zeros(D,E); k = zeros(n,E,D);
dMdm = zeros(E,D); dSdm = zeros(E,E,D); dCdm = zeros(D,E,D); 
dMds = zeros(E,D,D); dSds = zeros(E,E,D,D); dCds = zeros(D,E,D,D); 
dMdcen = zeros(E,n,D); dSdcen = zeros(E,E,n,D); dCdcen = zeros(D,E,n,D);
dMdll = zeros(E,D,E); dSdll = zeros(E,E,D,E); dCdll = zeros(D,E,D,E);
dMdw = zeros(E,n,E); dSdw = zeros(E,E,n,E); dCdw = zeros(D,E,n,E); 
r = zeros(1,D); T = zeros(D);Z = zeros(n,D); kdX = zeros(n,E,D);
QdXj = zeros(D);

if nargout < 4, [M, S, C] = conGauss(policy, m, s); return; end

inp = bsxfun(@minus,cen,m');                     % centralize training inputs

% 1) Predicted Mean and Input-Output Covariance *******************************
for i = 1:E
  
  % 1a) Compute the values ****************************************************
  RR = s+diag(exp(2*ll(:,i)));
  LL = diag(exp(-ll(:,i)));
  BB = LL*s*LL+eye(D); iRR = LL/BB*LL;
  tt = inp*iRR;
  l = exp(-sum(tt.*inp,2)/2); lb = l.*w(:,i);
  tlb = bsxfun(@times,tt,lb);
  c = 1/sqrt(det(RR))*exp(sum(ll(:,i)));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  C(:,i) = tt'*lb*c;                                  % input-output covariance
  v = bsxfun(@rdivide,inp,exp(ll(:,i)')); k(:,i) = -sum(v.*v,2)/2;
  
  % 1b) Compute the derivatives ***********************************************
  % ------------------------------------------------------- derivatives w.r.t s
  dMds(i,:,:) = c*tt'*tlb/2-iRR*M(i)/2;
  for d = 1:D
    dCds(d,i,:,:) = c*bsxfun(@times,tt,tt(:,d))'*tlb/2 - iRR*C(d,i)/2 ...
                                       - C(:,i)*iRR(d,:)/2 -iRR(:,d)*C(:,i)'/2;
  end
  
  % ---------------------------------- derivatives w.r.t basis-function weights
  dMdw(i,:,i) = c*l';
  dCdw(:,i,:,i) = c*bsxfun(@times,tt,l)';
  
  % ------------------------------------------- derivatives w.r.t length-scales
    detdX = diag(bsxfun(@times,det(RR)*iRR',2.*exp(2.*ll(:,i))));
    cdX = -0.5*c/det(RR).*detdX'+ c.*ones(1,D);
    dldX = bsxfun(@times,l,bsxfun(@times,tt,2.*exp(2*ll(:,i)')).*tt./2);
    dslb = zeros(1,D);
    for d = 1:D
      dslb(d) = w(:,i)'*dldX(:,d);
      dtdX = inp*(-bsxfun(@times,iRR(:,d),2.*exp(2*ll(d,i))*iRR(d,:)));
      dlbt = lb'*dtdX + (dldX(:,d).*w(:,i))'*tt;
      dCdll(:,i,d,i) = (dlbt'*c + cdX(d)*(lb'*tt)');
      kdX(:,i,d) = bsxfun(@times,v(:,d),v(:,d));
    end
    dMdll(i,:,i) = cdX.*sum(w(:,i).*l) + c.*dslb;
  
  % ----------------------------------------- derivatives w.r.t RBF centres
    dMdcen(i,:,:) = -c*tlb;
    for d = 1:D
      dCdcen(:,i,:,d) = c*(iRR(:,d)*lb' - bsxfun(@times,tt,tlb(:,d))');
    end
  
end % i

% --------------------------------------------------------- derivatives w.r.t m
dMdm = C';
dCdm = 2*permute(dMds,[2 1 3]);


% 2) Predictive Covariance Matrix *********************************************
for i = 1:E
  ii = bsxfun(@rdivide,inp,exp(2*ll(:,i)'));

  for j = 1:i
    
    % 2a) Compute the value ***************************************************
    R = s*diag(exp(-2*ll(:,i))+exp(-2*ll(:,j)))+eye(D); t = 1./sqrt(det(R));
    iR = R\eye(D);
    ij = bsxfun(@rdivide,inp,exp(2*ll(:,j)'));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    A = w(:,i)*w(:,j)'; A = A.*L; ssA = sum(sum(A));
    S(i,j) = t*ssA; S(j,i) = S(i,j);
    
    % 2b) Compute the derivatives *********************************************
    zzi = ii*(R\s); zzj = ij*(R\s);
    zi = ii/R; zj = ij/R;

    tdX = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*ll(:,i)')-2*exp(-2*ll(:,i)')));
    tdXi = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*ll(:,i)')));
    tdXj = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*ll(:,j)')));

    Q2 = R\s/2;
    aQ = ii*Q2; bQ = ij*Q2; 
    
    for d = 1:D
    % ----------------------------------------------- derivatives w.r.t m and s
      B = bsxfun(@plus,zi(:,d),zj(:,d)').*A;
      r(d) = sum(sum(B))*t;
      T(d,1:d) = sum(zi(:,1:d)'*B,2) + sum(B*zj(:,1:d))'; T(1:d,d) = T(d,1:d)';
      
    % --------------------------------------- derivatives w.r.t training inputs
      Z(:,d) = exp(-2*ll(d,i))*(A*zzj(:,d) + sum(A,2).*(zzi(:,d) - inp(:,d)))...
          + exp(-2*ll(d,j))*((zzi(:,d))'*A + sum(A,1).*(zzj(:,d) - inp(:,d))')';

    % ----------------------------------------- derivatives w.r.t length-scales
        if i==j
          RTi =  bsxfun(@times,s,-2*exp(-2*ll(:,i)')-2*exp(-2*ll(:,j)'));
          diRi = -R\bsxfun(@times,RTi(:,d),iR(d,:));
        else
          RTi = bsxfun(@times,s,-2*exp(-2*ll(:,i)'));
          RTj = bsxfun(@times,s,-2*exp(-2*ll(:,j)'));
          diRi = -R\bsxfun(@times,RTi(:,d),iR(d,:));
          diRj = -R\bsxfun(@times,RTj(:,d),iR(d,:));
          QdXj = diRj*s/2;
        end
      
        QdXi = diRi*s/2;
        
        if i==j
          daQi = ii*QdXi + bsxfun(@times,-2*ii(:,d),Q2(d,:));
          dsaQi = sum(daQi.*ii,2) - 2.*aQ(:,d).*ii(:,d); dsaQj = dsaQi;
          dsbQi = dsaQi; dsbQj = dsbQi;
          dm2i = -2*daQi*ii' + 2*(bsxfun(@times,aQ(:,d),ii(:,d)')...
                                +bsxfun(@times,ii(:,d),aQ(:,d)')); dm2j = dm2i;
        else
          dbQi = ij*QdXi;
          dbQj = ij*QdXj + bsxfun(@times,-2*ij(:,d),Q2(d,:));
          daQi = ii*QdXi + bsxfun(@times,-2*ii(:,d),Q2(d,:));
          daQj = ii*QdXj;

          dsaQi = sum(daQi.*ii,2) - 2.*aQ(:,d).*ii(:,d);
          dsaQj = sum(daQj.*ii,2);
          dsbQi = sum(dbQi.*ij,2);
          dsbQj = sum(dbQj.*ij,2) - 2.*bQ(:,d).*ij(:,d);
          dm2i = -2*daQi*ij';
          dm2j = -2*ii*(dbQj)';
        end
        
        dm1i = bsxfun(@plus,dsaQi,dsbQi');
        dm1j = bsxfun(@plus,dsaQj,dsbQj');
        dmahai = dm1i-dm2i; 
        dmahaj = dm1j-dm2j;
      
        if i==j
          LdXi = L.*(dmahai + bsxfun(@plus,kdX(:,i,d),kdX(:,j,d)'));
          dSdll(i,i,d,i) = w(:,i)'*LdXi*w(:,j);
        else
          LdXi = L.*(dmahai + bsxfun(@plus,kdX(:,i,d),zeros(n,1)'));
          LdXj = L.*(dmahaj + bsxfun(@plus,zeros(n,1),kdX(:,j,d)'));
          dSdll(i,j,d,i) = w(:,i)'*LdXi*w(:,j);
          dSdll(i,j,d,j) = w(:,i)'*LdXj*w(:,j);
        end
      
    end % d
    
    % -------------------------------------------------------------- bookeeping
    dSdm(i,j,:) = r;
    dSds(i,j,:,:) = (t*T-S(i,j)*diag(exp(-2*ll(:,i))+exp(-2*ll(:,j)))/R)/2;    
    dSdcen(i,j,:,:) = Z*t;
    if i==j
      dSdw(i,i,:,i) = w(:,i)'*(L+L')*t;
      dSdll(i,i,:,i) = reshape(t*dSdll(i,i,:,i),D,1)' + tdX*ssA;
    else
      dSdw(i,j,:,i) = w(:,j)'*L'*t;
      dSdw(i,j,:,j) = w(:,i)'*L *t;
      dSdll(i,j,:,i) = reshape(t*dSdll(i,j,:,i),D,1)' + tdXi*ssA;
      dSdll(i,j,:,j) = reshape(t*dSdll(i,j,:,j),D,1)' + tdXj*ssA;
    end
    
    % ------------------------------------------- centralise moment derivatives
    dSdm(i,j,:)   =shiftdim(dSdm(i,j,:)  ,1)-M(i)*dMdm(j,:)  -M(j)*dMdm(i,:);
    dSds(i,j,:,:) =shiftdim(dSds(i,j,:,:),1)-M(i)*dMds(j,:,:)-M(j)*dMds(i,:,:);
    dSdcen(i,j,:,:) =shiftdim(dSdcen(i,j,:,:),1)-M(i)*dMdcen(j,:,:)-M(j)*dMdcen(i,:,:);
    dSdw(i,j,:,i) =shiftdim(dSdw(i,j,:,i),1)-M(j)*dMdw(i,:,i);
    dSdw(i,j,:,j) =shiftdim(dSdw(i,j,:,j),1)-M(i)*dMdw(j,:,j);
    dSdll(i,j,:,i) =shiftdim(dSdll(i,j,:,i),1)-M(j)*dMdll(i,:,i);
    dSdll(i,j,:,j) =shiftdim(dSdll(i,j,:,j),1)-M(i)*dMdll(j,:,j);
    
    % ---------------------------------------------- fill in the symmetric bits
    if i~=j
      dSdm(j,i,:)   = dSdm(i,j,:);
      dSds(j,i,:,:) = dSds(i,j,:,:);
      dSdcen(j,i,:,:) = dSdcen(i,j,:,:);
      dSdw(j,i,:,:) = dSdw(i,j,:,:);
      dSdll(j,i,:,:) = dSdll(i,j,:,:);
    end
    
  end % j
end % i

S = S - M*M' + 1e-6*eye(E);               % centralise moments...and add jitter

% Vectorize the derivatives
dSdm = reshape(dSdm,E*E,D); dCdm = reshape(dCdm,D*E,D);
dMds = dMds(:,:);  dSds = reshape(dSds,E*E,D*D); dCds = reshape(dCds,D*E,D*D);

% Concatenate policy derivatives
dMdp = [dMdcen(:,:) dMdll(:,:) dMdw(:,:)];
dSdp = [reshape(dSdcen,E*E,n*D) reshape(dSdll,E*E,D*E) reshape(dSdw,E*E,n*E)];
dCdp = [reshape(dCdcen,D*E,n*D) reshape(dCdll,D*E,D*E) reshape(dCdw,D*E,n*E)];
