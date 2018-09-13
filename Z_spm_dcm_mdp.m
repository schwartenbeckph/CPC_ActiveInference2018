function [DCM] = Z_spm_dcm_mdp(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)
%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 2^-3;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'alpha')
            pE.(field) = log(16);               % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'beta')
            pE.(field) = log(1);                % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'cs')
            pE.(field) = log(2^0);              % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'cr')
            pE.(field) = log(2^2);              % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta')
            pE.(field) = log(0.5/(1-0.5));      % in logit-space - bounded between 0 and 1!
            pC{i,i}    = prior_variance;
        else
            pE.(field) = 0;      
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% multiply parameters in MDP
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'alpha')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'beta')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'cs')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'cr')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'eta')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    else
        mdp.(field{i}) = exp(P.(field{i}));
    end
end


% discern whether learning is enabled - and identify unique trials if not
%--------------------------------------------------------------------------
if any(ismember(fieldnames(mdp),{'a','b','d','c','d','e'}))
    j = 1:numel(U);
    k = 1:numel(U);
else
    % find unique trials (up until the last outcome)
    %----------------------------------------------------------------------
    u       = spm_cat(U');
    [i,j,k] = unique(u(:,1:(end - 1)),'rows');
end


% place MDP in trial structure
%--------------------------------------------------------------------------
n          = numel(j);

cs = 2^0; % standard preference for safe option
cr = 2^2; % standard preference for risky option win

if isfield(M.pE,'cs')&&isfield(M.pE,'cr')
    mdp.C{1}  = [0 mdp.cs mdp.cr -mdp.cs 0 0]';
elseif isfield(M.pE,'cs')
    mdp.C{1}  = [0 mdp.cs cr -mdp.cs 0 0]';
elseif isfield(M.pE,'cr')
    mdp.C{1}  = [0 cs mdp.cr -cs 0 0]';
else
    mdp.C{1}  = [0 cs cr -cs 0 0]';
end

[MDP(1:n)] = deal(mdp);
[MDP.o]    = deal(U{j});

% solve MDP and accumulate log-likelihood
%--------------------------------------------------------------------------
MDP   = Z_spm_MDP_VB_X(MDP);
L     = 0;
for i = 1:numel(Y)
    for j = 1:numel(Y{1})
        L = L + log(MDP(k(i)).P(Y{i}(j),j));
    end
end

