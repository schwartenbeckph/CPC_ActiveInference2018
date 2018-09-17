% Practical for Comp Psych in Zurich 2018
% Second Script: Computational Phenotyping

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part II %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

% Before you start, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder, where the practicals live, to your path in Matlab.

%% Set up model structure - again, same as in Practical_I

rng('default')

% Outcome probabilities: A
%==========================================================================

%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes.
%--------------------------------------------------------------------------

a = .75;
b = 1 - a;

A{1}   = [1 1 0 0 0 0 0 0;    % ambiguous starting position (centre)
          0 0 1 1 0 0 0 0;    % safe arm selected and rewarded
          0 0 0 0 a b 0 0;    % risky arm selected and rewarded
          0 0 0 0 b a 0 0;    % risky arm selected and not rewarded
          0 0 0 0 0 0 1 0;    % informative cue - high reward prob
          0 0 0 0 0 0 0 1];   % informative cue - low reward prob
 
% Controlled transitions: B{u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% under each action or control state. Here, there are four actions taking the
% agent directly to each of the four locations.
%--------------------------------------------------------------------------

% move to/stay in the middle
B{1}(:,:,1) = [1 0 0 0 0 0 1 0;
               0 1 0 0 0 0 0 1;
               0 0 1 0 0 0 0 0;
               0 0 0 1 0 0 0 0;
               0 0 0 0 1 0 0 0;
               0 0 0 0 0 1 0 0;
               0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0];
           
% move up left to safe  (and check for reward)           
B{1}(:,:,2) = [0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0;
               1 0 1 0 0 0 1 0;
               0 1 0 1 0 0 0 1;
               0 0 0 0 1 0 0 0;
               0 0 0 0 0 1 0 0;
               0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0];

% move up right to risky (and check for reward)           
B{1}(:,:,3) = [0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0;
               0 0 1 0 0 0 0 0;
               0 0 0 1 0 0 0 0;
               1 0 0 0 1 0 1 0;
               0 1 0 0 0 1 0 1;
               0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0];

% move down (check cue)           
B{1}(:,:,4) = [0 0 0 0 0 0 0 0;
               0 0 0 0 0 0 0 0;
               0 0 1 0 0 0 0 0;
               0 0 0 1 0 0 0 0;
               0 0 0 0 1 0 0 0;
               0 0 0 0 0 1 0 0;
               1 0 0 0 0 0 1 0;
               0 1 0 0 0 0 0 1];           
          
% Priors: 
%==========================================================================

%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities. Here, the agent prefers rewarding outcomes
%--------------------------------------------------------------------------
cs = 2^0; % preference for safe option
cr = 2^2; % preference for risky option win

% preference for: [staying at starting point | safe | risky + reward | risky + no reward | cue context 1 | cue context 2]
C{1}  = [0 cs cr -cs 0 0]';

%--------------------------------------------------------------------------
% Now specify prior beliefs about initial state
%--------------------------------------------------------------------------
D{1}  = kron([1/4 0 0 0],[1 1])';

% Allowable policies (of depth T).  These are sequences of actions
%==========================================================================

V  = [1  1  1  1  2  3  4  4  4  4
      1  2  3  4  2  3  1  2  3  4];
 
 
%% Define MDP Structure - again
%==========================================================================
%==========================================================================

mdp.V = V;                    % allowable policies
mdp.A = A;                    % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % prior over initial states
mdp.s = 1;                    % initial state

mdp.eta = 0.5;                % learning rate


%--------------------------------------------------------------------------
% Check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);
 
%--------------------------------------------------------------------------
% Having specified the basic generative model, let's see how active
% infernece solves different tasks
%--------------------------------------------------------------------------


%% 6. Model inversion to recover parameters (preferences and precision)
%==========================================================================
%==========================================================================

close all

%% 6.1 Produce IMprecise behaviour WITH information-gain (again):
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP);
[MDP(i).s] = deal(2);


% true parameter values:
[MDP(1:n).alpha] = deal(4);              % precision of action selection - this is smaller than prior mean, check Z_spm_dcm_mdp

MDP  = Z_spm_MDP_VB_X(MDP);

%% 6.2 Invert model and try to recover original parameters:
%==========================================================================

%--------------------------------------------------------------------------
% This is the model inversion part. Model inversion is based on variational
% Bayes. The basic idea is to maximise (negative) variational free energy
% wrt to the free parameters (here: alpha and cr). This means maximising
% the likelihood of the data under these parameters (i.e., maximise
% accuracy) and at the same time penalising for strong deviations from the
% priors over the parameters (i.e., minimise complexity), which prevents
% overfitting.
% 
% You can specify the prior mean and variance of each parameter at the
% beginning of the Z_spm_dcm_mdp script.
%--------------------------------------------------------------------------

DCM.MDP   = mdp;                  % MDP model
DCM.field = {'alpha','cr'};       % parameter (field) names to optimise
DCM.U     = {MDP.o};              % trial specification (stimuli)
DCM.Y     = {MDP.u};              % responses (action)
 
DCM       = Z_spm_dcm_mdp(DCM);
 
subplot(2,2,3)
xticklabels(DCM.field),xlabel('Parameter')
subplot(2,2,4)
xticklabels(DCM.field),xlabel('Parameter')
 
%% 6.3 Check deviation of prior and posterior means & posterior covariance:
%==========================================================================

%--------------------------------------------------------------------------
% re-transform values and compare prior with posterior estimates
%--------------------------------------------------------------------------

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
    else
        prior(i) = exp(DCM.M.pE.(field{i}));
        posterior(i) = exp(DCM.Ep.(field{i}));
    end
end

figure, set(gcf,'color','white')
subplot(2,1,1),hold on
title('Means')
bar(prior,'FaceColor',[.5,.5,.5]),bar(posterior,0.5,'k')
xlim([0,length(prior)+1]),set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
legend({'Prior','Posterior'})
hold off
subplot(2,1,2)
imagesc(DCM.Cp),caxis([0 1]),colorbar
title('(Co-)variance')
set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
set(gca, 'YTick', 1:length(prior)),set(gca, 'YTickLabel', DCM.field)
 

%% 7. Now repeat using subsets of trials to illustrate effects on estimators - design optimisation!
%==========================================================================
%==========================================================================

DCM.MDP   = mdp;                  % MDP model
DCM.field = {'alpha'};

n         = [2 4 8 16 32];

for i = 1:length(n)
    
    DCM.U = {MDP(1:n(i)).o};
    DCM.Y = {MDP(1:n(i)).u};
    DCM   = Z_spm_dcm_mdp(DCM);
    
    Ep(i,1) = DCM.Ep.alpha;
    Cp(i,1) = DCM.Cp;
    
    fprintf('### Simulated parameter recovery with %d trials ###\n',n(i))
    
end
 
% plot results
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 3'); clf
subplot(2,1,1), spm_plot_ci(exp(Ep(:)),Cp(:)), hold on
plot(1:length(n),(n - n) + MDP(1).alpha,'k'),hold off
set(gca,'XTickLabel',n)
xlabel('number of trials','FontSize',12)
ylabel('conditional estimate','FontSize',12)
title('Dependency on trial number','FontSize',16)
axis square
 

%% 8. Now repeat but over multiple subjects with different alpha
%==========================================================================
%==========================================================================
 
% generate data and a between subject model with two groups of eight
% subjects
%--------------------------------------------------------------------------
N     = 8;                             % numbers of subjects per group
X     = kron([1 1;1 -1],ones(N,1));    % design matrix
h     = 4;                             % between subject log precision
n     = 32;                            % number of trials
i     = rand(1,n) > 1/2;               % randomise hidden states
 
clear MDP

[MDP(1:n)]       = deal(mdp);
[MDP(i).s]       = deal(2);

reward = zeros(n,size(X,1));
 
for i = 1:size(X,1)
   
    % true parameters - with a group difference of one
    %----------------------------------------------------------------------  
    alpha(i)    = X(i,:)*[0; 1] + exp(-h/2)*randn;         % add random Gaussian effects to group means
    [MDP.alpha] = deal(exp(alpha(i)));
 
    % solve to generate data
    %----------------------------------------------------------------------
    DDP        = Z_spm_MDP_VB_X(MDP);      % realisation for this subject
    
    DCM.field  = {'alpha'};
    DCM.U      = {DDP.o};              % trial specification (stimuli) 
    DCM.Y      = {DDP.u};              % responses (action)
    GCM{i,1}   = DCM;
   
    for kk=1:length(DCM.U)
        if DCM.U{kk}(end)==2 || DCM.U{kk}(end)==4 % outcome 2 or 4 == reward
            reward(kk,i)=1;
        end
    end
    
    % plot behavioural responses
    %----------------------------------------------------------------------
    if i == 1 || i == 9
        Z_spm_MDP_VB_game_EpistemicLearning_state(DDP);drawnow
    end
    
    fprintf('### Simulated data for subject %d of %d ###\n',i,size(X,1))
   
end
 
 
%% 9. Bayesian model inversion
%==========================================================================
%==========================================================================

%--------------------------------------------------------------------------
% Invert models of all 16 subjects - this will take a bit of time
%--------------------------------------------------------------------------

GCM  = Z_spm_dcm_fit(GCM);
 
% plot subject specific estimates and true values
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 4');
subplot(3,1,3)

for i = 1:length(GCM)
    qP(i) = GCM{i}.Ep.alpha;
end

plot(alpha,alpha,':b',alpha,qP,'.b','MarkerSize',32)
xlabel('true parameter','FontSize',12)
ylabel('conditional estimate','FontSize',12)
title('Subject specific estimates','FontSize',16)
axis square
 
                    
%% 10. hierarchical Bayes
%==========================================================================
%==========================================================================

%--------------------------------------------------------------------------
% Using PEB, you can test for the evidence of a 'full' model that assumes a
% group difference in all parameters and simpler models that assume no
% differences.
% 
% This allows to test for evidence for a difference (or for no difference)
% in estimated parameters. See relevant literature on these routines, e.g. 
% Friston, Litvak, Oswal, Razi, Stephan, van Wijk, Ziegler, & Zeidman, 2016
%--------------------------------------------------------------------------

% second level model
%--------------------------------------------------------------------------
M    = struct('X',X);
 
% BMA - (second level)
%--------------------------------------------------------------------------
PEB  = spm_dcm_peb(GCM,M);
BMA  = spm_dcm_peb_bmc(PEB);
 
subplot(3,2,1), set(gca,'XTickLabel',DCM.field), title('Mean'), ylim([-1,3])
subplot(3,2,2), set(gca,'XTickLabel',DCM.field), title('Group Effect'),ylim([-1,3])
subplot(3,2,3), set(gca,'XTickLabel',DCM.field), ylim([-1,3])
subplot(3,2,4), set(gca,'XTickLabel',DCM.field), ylim([-1,3])
subplot(3,2,5), set(gca,'XTickLabel',DCM.field), ylim([0,1.2])
subplot(3,2,6), set(gca,'XTickLabel',DCM.field), ylim([0,1.2])

 
%% 11. posterior predictive density and cross validation
%==========================================================================
%==========================================================================

%--------------------------------------------------------------------------
% This is leave-one-out crossvalidation to test whether the estimated
% parameter can be used to infer group membership.
%--------------------------------------------------------------------------

spm_dcm_loo(GCM,M,'alpha');

%--------------------------------------------------------------------------
% This completes the overview over computational phenotyping in active inference. 
% We now turn to a final example, illustrating active learning as opposed
% to active inference as described in part I
%-------------------------------------------------------------------------