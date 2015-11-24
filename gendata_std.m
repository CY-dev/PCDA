function [X,y,beta,A] = gendata_std(n,p,K,sigma,ratio,seednum,rho)
%========================================================================%
% INPUTS:                                                                %
%         n   ---- number of samples                                     %
%         p   ---- signal length                                         %
%         K   ---- number of nonzero elements in the signal              %
%      ratio  ---- range of value in the signal (= 10^ratio)             %
%      sigma  ---- noise variance                                        %
%     seednum ---- seed number                                           %
%        rho  ---- coorelation                                           %
% OUTPUTS:                                                               %
%     X    ---- standardized design matrix                               %
%     y    ---- response vector with noise                               %
%     beta ---- true signal                                              %
%     A    ---- the support of beta                                      % 
%------------------------------------------------------------------------%
% Created by Congrui Yi (congrui-yi@uiowa.edu) on Sep 28, 2015           % 
% Modified from 'gendata' created by                                     %
%    Yuling Jiao (yulingjiaomath@whu.edu.cn)                             %
%    and Bangti Jin  (bangti.jin@gmail.com)                              %
%========================================================================%
disp('Generating data...')
rand('seed',seednum);   % fix seed
randn('seed', seednum); % fix seed
% generate signal
beta = zeros(p,1);     % true signal  
q = randperm(p);
A = q(1:K);
if ratio ~= 0
    v = rand(K,1);
    v = v-min(v);
    v = v/max(v)*ratio;
    beta(A) = 10.^v.*sign(randn(K,1));
else
    beta(A) = sign(randn(K,1));
end

% generate matrix X
% creat X 
X = randn(n,p);
if rho ~= 0
    SX = zeros(p,p);
for k = 1:p
    for l = 1:p 
        SX(k,l) = rho^(abs(k-l));
    end
end
X= X*chol(SX);
end
X = zscore(X); % standardize
% generate response
ye = X*beta;
y = ye + sigma*randn(n,1);
A = find(beta);
disp('Done')
end