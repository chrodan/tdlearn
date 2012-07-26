function [M, S, V] = gpP0(X, input, target, m, s)

% Compute joint predictions for multiple GPs with uncertain inputs. The
% predictive variance contains uncertainty about the function, but no noise.
%
% X       E*(D+2) by 1 vector of log-hyper-parameters
% input   n by D matrix of training inputs
% target  n by E matrix of training targets
% m       D by 1 vector, mean of the test distribution
% s       D by D covariance matrix of the test distribution
%
% M       E by 1 vector, mean of pred. distribution
% S       E by E matrix, covariance of the pred. distribution
% V       D by E inv(s) times covariance between input and prediction
%
% Copyright (C) 2008-2011 by Carl Edward Rasmussen & Marc Deisenroth,
% 2011-05-04

persistent K iK oldX;
[n, D] = size(input);         % number of examples and dimension of input space
[n, E] = size(target);               % number of examples and number of outputs
X = reshape(X, D+2, E)';

% if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) || sum(any(X ~= oldX))
  oldX = X;                                               
  iK = zeros(n,n,E); K = zeros(n,n,E);
  
  for i=1:E
    inp = bsxfun(@rdivide,input,exp(X(i,1:D)));
    K(:,:,i) = exp(2*X(i,D+1)-maha(inp,inp)/2);  % compute K and inv(K)
    L = chol(K(:,:,i)+exp(2*X(i,D+2))*eye(n))';
    iK(:,:,i) = L'\(L\eye(n));
  end
end

k = zeros(n,E); beta = k; M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,input,m');                         % centralize inputs

for i = 1:E     % compute predicted mean and inv(s) times input-output covariance
  Lambda = diag(exp(-X(i,1:D)));
  in = inp*Lambda;
  B = Lambda*s*Lambda+eye(D); 
  beta(:,i) = (K(:,:,i)+exp(2*X(i,D+2))*eye(n))\target(:,i);  
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tLambda = t*Lambda;
  c = exp(2*X(i,D+1))/sqrt(det(B));
  M(i) = c*sum(lb);
  V(:,i) = tLambda'*lb*c;                   % inv(s) times input-output covariance
  k(:,i) = 2*X(i,D+1)-sum(in.*in,2)/2;
end

for i=1:E                  % compute predictive covariance, non-central moments
  ii = bsxfun(@rdivide,inp,exp(2*X(i,1:D)));
  
  for j=1:i
    R = s*diag(exp(-2*X(i,1:D))+exp(-2*X(j,1:D)))+eye(D); 
    t = 1/sqrt(det(R));
    ij = bsxfun(@rdivide,inp,exp(2*X(j,1:D)));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    A = beta(:,i)*beta(:,j)'; 
    
    if i==j; A = A - iK(:,:,i); end;            % incorporate model uncertainty
    
    A = A.*L;
    S(i,j) = t*sum(sum(A)); S(j,i) = S(i,j);   
  end
  
  S(i,i) = S(i,i) + exp(2*X(i,D+1));
end

S = S - M*M';                                              % centralize moments
