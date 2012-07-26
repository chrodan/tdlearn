function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdi, dSdi, dVdi, ...
  dMdt, dSdt, dVdt, dMdX, dSdX, dVdX] = gpP2d(X, input, target, m, s)
%
% Compute joint predictions and derivatives for GPs with uncertain inputs.
%
% inputs:
% X       E*(D+2) by 1 vector of log-hyper-parameters
% input   n by D matrix of training inputs
% target  n by E matrix of training targets
% m       D by 1 vector, mean of the test distribution
% s       D by D covariance matrix of the test distribution
%
% outputs:
% M       E by 1 vector, mean of pred. distribution
% S       E by E matrix, covariance of the pred. distribution
% V       D by E inv(s) times covariance between input and prediction
% dMdm    E by D matrix of output mean by input mean partial derivatives
% dSdm    E by E by D matrix of output covariance by input mean derivatives
% dVdm    D by E by D matrix of inv(s) times input-output cov by input mean derivatives
% dMds    E by D by D matrix of ouput mean by input covariance derivatives
% dSds    E by E by D by D matrix of output cov by input covariance derivatives
% dVds    D by E by D by D matrix of inv(s) times input-output covariance by input covariance

% dMdi    E by n by D matrix of output mean by training input derivatives
% dSdi    E by E by n by D matrix of output covariance by training input derivatives
% dVdi    D by E by n by E matrix of inv(s) times input-output covariance by training input derivatives
% dMdt    E by n by E matrix of output mean by training target derivatives
% dSdt    E by E by n by E matrix of output covariance by target derivatives
% dVdt    D by E by n by E matrix of inv(s) times input-output covariance by target derivatives
% dMdX    E by (D+2)*E matrix of output mean by hyper-parameters derivatives
% dSdX    E by E by (D+2)*E matrix of output covariance by hyper-parameters derivatives
% dVdX    D by E by (D+2)*E matrix of inv(s) times input output covariance by hyper-parameters derivatives
%
% compute
% E[p(f(x)|m,s)]
% S[p(f(x)|m,s)]
% cov(x,f(x)|m,s)
%
% does NOT include
% a) uncertainty about the underlying function (in prediction)
% b) measurement/system noise in the predictive covariance
%
%
% Copyright (C) 2008-2010 by Marc Deisenroth and Carl Edward Rasmussen,
% 2010-12-02

if nargout < 4; [M, S, V] = gpP0d(X, input, target, m, s); return; end

persistent K iK oldX oldIn;           % cache some variables
[n, D] = size(input);                 % number of examples and dimension of input space
[n, E] = size(target);                % number of examples and number of outputs
X = reshape(X, D+2, E)';

% if necessary: re-compute cashed variables
if length(X) ~= length(oldX) || isempty(iK) || ...
    sum(any(X ~= oldX)) || sum(any(oldIn ~= input))
  oldX = X; oldIn = input;
  K = zeros(n,n,E); iK = K;
  
  % compute K and inv(K)
  for i=1:E
    inp = bsxfun(@rdivide,input,exp(X(i,1:D)));
    K(:,:,i) = exp(2*X(i,D+1)-maha(inp,inp)/2);
    L = chol(K(:,:,i)+exp(2*X(i,D+2))*eye(n))';
    iK(:,:,i) = L'\(L\eye(n));
  end
  
end

% initializations
k = zeros(n,E); beta = k; M = zeros(E,1); V = zeros(D,E); S = zeros(E);
dMdm = zeros(E,D); dMds = zeros(E,D,D); dSdm = zeros(E,E,D); r = zeros(1,D);
dSds = zeros(E,E,D,D); dVdm = zeros(D,E,D); dVds = zeros(D,E,D,D); T = zeros(D);
tlbdi = zeros(n,D); dMdi = zeros(E,n,D); dMdt = zeros(E,n,E); dK = zeros(n,n,D);
dVdt = zeros(D,E,n,E); dVdi = zeros(D,E,n,D); dSdt = zeros(E,E,n,E);
dSdi = zeros(E,E,n,D); dMdX = zeros(E,D+2,E); dSdX = zeros(E,E,D+2,E);
dVdX = zeros(D,E,D+2,E); Z = zeros(n,D);
bdX = zeros(n,E,D); kdX = zeros(n,E,D+1);

% centralize training inputs
inp = bsxfun(@minus,input,m');

% 1) compute predicted mean and input-output covariance
for i = 1:E
  % first some useful intermediate terms
  K2 = K(:,:,i)+exp(2*X(i,D+2))*eye(n); % K + sigma^2*I
  inp2 = bsxfun(@rdivide,input,exp(X(i,1:D)));
  ii = bsxfun(@rdivide,input,exp(2*X(i,1:D)));
  beta(:,i) = (K2)\target(:,i);
%   iLambda = diag(exp(-2*X(i,1:D)));  % inverse squared length-scales
  R = s+diag(exp(2*X(i,1:D)));
%   iR = iLambda*(eye(D) - (eye(D)+s*iLambda)\(s*iLambda)); iR = (iR+iR')/2; % Kailath inverse
  L = diag(exp(-X(i,1:D)));
  B = L*s*L+eye(D); iR = L/B*L;
  t = inp*iR;
  l = exp(-sum(t.*inp,2)/2); lb = l.*beta(:,i);
  tliK = t'*bsxfun(@times,l,iK(:,:,i));
  liK = K2\l;
  tlb = bsxfun(@times,t,lb);
  
  c = exp(2*X(i,D+1))/sqrt(det(R))*exp(sum(X(i,1:D)));
  detdX = diag(bsxfun(@times,det(R)*iR',2.*exp(2.*X(i,1:D)))); % d(det R)/dX
  cdX = -0.5*c/det(R).*detdX'+ c.*ones(1,D);
  dldX = bsxfun(@times,l,bsxfun(@times,t,2.*exp(2*X(i,1:D))).*t./2); % der. wrt. length-scales
  
  M(i) = sum(lb)*c;                                        % predicted mean
  
  dMds(i,:,:) = c*t'*tlb/2-iR*M(i)/2;
  dMdX(i,D+2,i) = -c*sum(l.*(2*exp(2*X(i,D+2))*(K2\beta(:,i)))); % OK
  dMdX(i,D+1,i) = -dMdX(i,(i-1)*(D+2)+D+2);
  
  dVdX(:,i,D+2,i) = -((l.*(2*exp(2*X(i,D+2))*...
    ((K2)\beta(:,i))))'*t*c)';
  dVdX(:,i,D+1,i) = -dVdX(:,i,D+2,i);
  
  dsi = -bsxfun(@times,inp2,2.*inp2); % d(sum(inp2.*inp2,2))/dX
  dslb = zeros(1,D);
  
  for d = 1:D
    sqdi = K(:,:,i).*bsxfun(@minus,ii(:,d),ii(:,d)');
    tlbdi(:,d) = sqdi*liK.*beta(:,i) + sqdi*beta(:,i).*liK;
    tlbdi2 = -tliK*(-bsxfun(@times,sqdi,beta(:,i))'-diag(sqdi*beta(:,i)));
    dVdi(:,i,:,d) = c*(iR(:,d)*lb' - bsxfun(@times,t,tlb(:,d))' + tlbdi2);
    dsqdX = bsxfun(@plus,dsi(:,d),dsi(:,d)') + 4.*inp2(:,d)*inp2(:,d)';
    dK(:,:,d) = -K(:,:,i).*dsqdX./2;           % dK/dX(1:D)
    bdX(:,i,d) = -(K2)\(dK(:,:,d)*beta(:,i));  % dbeta/dX
    dslb(d) = -liK'*dK(:,:,d)*beta(:,i) + beta(:,i)'*dldX(:,d);
    dlb = dldX(:,d).*beta(:,i) - l.*((K2)\(dK(:,:,d)*beta(:,i)));
    dtdX = inp*(-bsxfun(@times,iR(:,d),2.*exp(2*X(i,d))*iR(d,:)));
    dlbt = lb'*dtdX + dlb'*t;
    dVdX(:,i,d,i) = (dlbt'*c + cdX(d)*(lb'*t)');
  end % d
  
  dMdi(i,:,:) = c*(tlbdi - tlb);
  dMdt(i,:,i) = c*liK';
  dMdX(i,1:D,i) = cdX.*sum(beta(:,i).*l) + c.*dslb;
  v = bsxfun(@rdivide,inp,exp(X(i,1:D)));
  k(:,i) = 2*X(i,D+1)-sum(v.*v,2)/2;
  V(:,i) = t'*lb*c;                               % input-output covariance
  
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,t,t(:,d))'*tlb/2 - iR*V(d,i)/2 ...
      - V(:,i)*iR(d,:)/2 -iR(:,d)*V(:,i)'/2;
    %     dVds(d,i,:,:) = ...
    %       c*bsxfun(@times,t,t(:,d))'*tlb/2 - iR*V(d,i)/2 - V(:,i)*iR(d,:);
    kdX(:,i,d) = bsxfun(@times,v(:,d),v(:,d));
  end % d
  
  dVdt(:,i,:,i) = c*tliK;
  kdX(:,i,D+1) = 2*ones(1,n);  % pre-computation for later
  
end % i
dMdm = V';                                              % derivatives wrt m
dVdm = 2*permute(dMds,[2 1 3]);


% 2) predictive covariance matrix
% 2a) non-central moments
for i = 1:E
  K2 = K(:,:,i)+exp(2*X(i,D+2))*eye(n);
  ii = bsxfun(@rdivide,inp,exp(2*X(i,1:D)));
  
  for j = 1:i % if i==j: diagonal elements of S; see Marc's thesis around eq. (2.26)
    R = s*diag(exp(-2*X(i,1:D))+exp(-2*X(j,1:D)))+eye(D); t = 1./sqrt(det(R));
    iR = R\eye(D);
    ij = bsxfun(@rdivide,inp,exp(2*X(j,1:D)));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2)); % called Q in thesis
    A = beta(:,i)*beta(:,j)'; A = A.*L; ssA = sum(sum(A));
    S(i,j) = t*ssA; S(j,i) = S(i,j);
    
    zzi = ii*(R\s);
    zzj = ij*(R\s);
    zi = ii/R; zj = ij/R;
    
    tdX = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*X(i,1:D))-2*exp(-2*X(i,1:D))));
    tdXi = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*X(i,1:D))));
    tdXj = -0.5*t*sum(iR'.*bsxfun(@times,s,-2*exp(-2*X(j,1:D))));
    bLiKi = iK(:,:,j)*(L'*beta(:,i)); bLiKj = iK(:,:,i)*(L*beta(:,j));
    Q2 = R\s/2;
    aQ = ii*Q2; bQ = ij*Q2;
    
    for d = 1:D
      %       Z(:,d) = exp(-2*X(i,d))*sum(A.*bsxfun(@plus,zzi(:,d)-inp(:,d),zzj(:,d)'),2)...
      %         +exp(-2*X(j,d))*sum(A.*bsxfun(@plus,zzi(:,d),zzj(:,d)'-inp(:,d)'),1)';
      
      Z(:,d) = exp(-2*X(i,d))*(A*zzj(:,d) + sum(A,2).*(zzi(:,d) - inp(:,d)))...
        + exp(-2*X(j,d))*((zzi(:,d))'*A + sum(A,1).*(zzj(:,d) - inp(:,d))')';
      Q = bsxfun(@minus,inp(:,d),inp(:,d)');
      B = K(:,:,i).*Q;
      Z(:,d) = Z(:,d)+exp(-2*X(i,d))*(B*beta(:,i).*bLiKj+beta(:,i).*(B*bLiKj));
      
      if i~=j; B = K(:,:,j).*Q; end
      
      Z(:,d) = Z(:,d)+exp(-2*X(j,d))*(bLiKi.*(B*beta(:,j))+B*bLiKi.*beta(:,j));
      B = bsxfun(@plus,zi(:,d),zj(:,d)').*A;
      r(d) = sum(sum(B))*t;
      T(d,1:d) = sum(zi(:,1:d)'*B,2) + sum(B*zj(:,1:d))';
      T(1:d,d) = T(d,1:d)';
      
      if i==j
        RTi =  bsxfun(@times,s,(-2*exp(-2*X(i,1:D))-2*exp(-2*X(j,1:D))));
        diRi = -R\bsxfun(@times,RTi(:,d),iR(d,:));
      else
        RTi = bsxfun(@times,s,-2*exp(-2*X(i,1:D)));
        RTj = bsxfun(@times,s,-2*exp(-2*X(j,1:D)));
        diRi = -R\bsxfun(@times,RTi(:,d),iR(d,:));
        diRj = -R\bsxfun(@times,RTj(:,d),iR(d,:));
        QdXj = diRj*s/2; % dQ2/dXj
      end
      
      QdXi = diRi*s/2; % dQ2/dXj
      
      if i==j
        daQi = ii*QdXi + bsxfun(@times,-2*ii(:,d),Q2(d,:)); % d(ii*Q)/dXi
        dsaQi = sum(daQi.*ii,2) - 2.*aQ(:,d).*ii(:,d); dsaQj = dsaQi;
        dsbQi = dsaQi; dsbQj = dsbQi;
        dm2i = -2*daQi*ii' + 2*(bsxfun(@times,aQ(:,d),ii(:,d)')...
          +bsxfun(@times,ii(:,d),aQ(:,d)')); dm2j = dm2i; % -2*aQ*ij'/di
      else
        dbQi = ij*QdXi;  % d(ij*Q)/dXi
        dbQj = ij*QdXj + bsxfun(@times,-2*ij(:,d),Q2(d,:)); % d(ij*Q)/dXj
        daQi = ii*QdXi + bsxfun(@times,-2*ii(:,d),Q2(d,:)); % d(ii*Q)/dXi
        daQj = ii*QdXj; % d(ii*Q)/dXj
        
        dsaQi = sum(daQi.*ii,2) - 2.*aQ(:,d).*ii(:,d);
        dsaQj = sum(daQj.*ii,2);
        dsbQi = sum(dbQi.*ij,2);
        dsbQj = sum(dbQj.*ij,2) - 2.*bQ(:,d).*ij(:,d);
        dm2i = -2*daQi*ij'; % second part of the maha(..) function wrt Xi
        dm2j = -2*ii*(dbQj)'; % second part of the maha(..) function wrt Xj
      end
      
      dm1i = bsxfun(@plus,dsaQi,dsbQi'); % first part of the maha(..) function wrt Xi
      dm1j = bsxfun(@plus,dsaQj,dsbQj'); % first part of the maha(..) function wrt Xj
      dmahai = dm1i-dm2i;
      dmahaj = dm1j-dm2j;
      
      if i==j
        LdXi = L.*(dmahai + bsxfun(@plus,kdX(:,i,d),kdX(:,j,d)'));
        dSdX(i,i,d,i) = beta(:,i)'*LdXi*beta(:,j);
      else
        LdXi = L.*(dmahai + bsxfun(@plus,kdX(:,i,d),zeros(n,1)'));
        LdXj = L.*(dmahaj + bsxfun(@plus,zeros(n,1),kdX(:,j,d)'));
        dSdX(i,j,d,i) = beta(:,i)'*LdXi*beta(:,j);
        dSdX(i,j,d,j) = beta(:,i)'*LdXj*beta(:,j);
      end
      
    end % d
    
    if i==j
      dSdX(i,i,1:D,i) = reshape(dSdX(i,i,1:D,i),D,1) + reshape(bdX(:,i,:),n,D)'*(L+L')*beta(:,i);
      dSdX(i,i,1:D,i) = reshape(t*dSdX(i,i,1:D,i),D,1)' + tdX*ssA;
      dSdX(i,i,D+2,i) = 2*exp(2*X(i,D+2))*t*(-sum(beta(:,i).*bLiKi)-sum(beta(:,i).*bLiKi));
    else
      dSdX(i,j,1:D,i) = reshape(dSdX(i,j,1:D,i),D,1) + reshape(bdX(:,i,:),n,D)'*(L*beta(:,j));
      dSdX(i,j,1:D,j) = reshape(dSdX(i,j,1:D,j),D,1) + reshape(bdX(:,j,:),n,D)'*(L'*beta(:,i));
      dSdX(i,j,1:D,i) = reshape(t*dSdX(i,j,1:D,i),D,1)' + tdXi*ssA;
      dSdX(i,j,1:D,j) = reshape(t*dSdX(i,j,1:D,j),D,1)' + tdXj*ssA;
      dSdX(i,j,D+2,i) = 2*exp(2*X(i,D+2))*t*(-beta(:,i)'*bLiKj);
      dSdX(i,j,D+2,j) = 2*exp(2*X(j,D+2))*t*(-beta(:,j)'*bLiKi);
    end
    
    dSdm(i,j,:) = r - M(i)*dMdm(j,:)-M(j)*dMdm(i,:); dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S(i,j)*diag(exp(-2*X(i,1:D))+exp(-2*X(j,1:D)))/R)/2;
    T = T - reshape(M(i)*dMds(j,:,:) + M(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T; 
    dSds(j,i,:,:) = T;
    
    if i==j
      dSdt(i,i,:,i) = (beta(:,i)'*(L+L'))/(K2)*t ...
        - 2*dMdt(i,:,i)*M(i);
      dSdX(i,j,:,i) = reshape(dSdX(i,j,:,i),1,D+2) - M(i)*dMdX(j,:,j)-M(j)*dMdX(i,:,i);
    else
      dSdt(i,j,:,i) = (beta(:,j)'*L')/(K2)*t ...
        - dMdt(i,:,i)*M(j);
      dSdt(i,j,:,j) = beta(:,i)'*L/(K(:,:,j)+exp(2*X(j,D+2))*eye(n))*t ...
        - dMdt(j,:,j)*M(i);
      dSdt(j,i,:,:) = dSdt(i,j,:,:);
      dSdX(i,j,:,j) = reshape(dSdX(i,j,:,j),1,D+2) - M(i)*dMdX(j,:,j);
      dSdX(i,j,:,i) = reshape(dSdX(i,j,:,i),1,D+2) - M(j)*dMdX(i,:,i);
    end
    
    dSdi(i,j,:,:) = Z*t - reshape(M(i)*dMdi(j,:,:) + dMdi(i,:,:)*M(j),n,D);
    dSdi(j,i,:,:) = dSdi(i,j,:,:);
    dSdX(j,i,:,:) = dSdX(i,j,:,:);
  end % j
  
  S(i,i) = S(i,i) + 1e-06;  % add small diagonal jitter for numerical reasons
end % i

dSdX(:,:,D+1,:) = -dSdX(:,:,D+2,:);
dSdX(:,:,D+1,:) = -dSdX(:,:,D+2,:);

% 2b) centralize moments
S = S - M*M';

% reshape the d*/dX variables to make them consistent with input X
dMdX = reshape(dMdX,E,(D+2)*E);
dVdX = reshape(dVdX,D,E,(D+2)*E);
dSdX = reshape(dSdX,E,E,(D+2)*E);