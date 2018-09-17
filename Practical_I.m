% Practical for Comp Psych in Zurich 2018
% First Script: Model a maze task that illustrates trade-off between
% information gain and maximising reward

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part I %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

% Before you start, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder, where the practicals live, to your path in Matlab.

%--------------------------------------------------------------------------
% This routine uses a Markov decision process formulation of active
% inference (with variational Bayes) to model foraging for information in a
% three arm maze.  This demo illustrates the inversion of a single subject
% and group data to make inferences about subject specific parameters –
% such as their prior beliefs about precision and utility.
%
% We first generate some synthetic data for a single subject and illustrate
% the recovery of key parameters using variational Laplace. We then
% consider the inversion of multiple trials from a group of subjects to
% illustrate the use of empirical Bayes in making inferences at the between
% subject level – and the use of Bayesian cross-validation to retrieve out
% of sample estimates (and classification of new subjects)
%
% In this example, the agent starts at the centre of a three way maze with 
% a safe (left) and a risky (right) option, where the risky option either 
% has a high (75%) or low (25%) reward probability. However, the reward 
% probability changes from trial to trial.  Crucially, the agent can 
% identify the current reward probability by accessing a cue in the lower 
% arm. This tells the agent whether the reward probability of this trial is
% high or low.  
% This means the optimal policy would first involve maximising information 
% gain (epistemic value) by moving to the lower arm and then choosing the
% safe or the risky option. Here, there are eight hidden states 
% (four locations times high or low reward context), four control states
% (that take the agent to the four locations) and seven outcomes (three
% locations times two cues plus the centre).  
% 
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging
%--------------------------------------------------------------------------
 
%% 1. Set up model structure
%==========================================================================
%==========================================================================

rng('default')


%% 1.1 Outcome probabilities: A
%==========================================================================

%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes.
% 
% Columns reflect hidden states, while rows reflect observations. Entries
% reflect probabilities for making an observation, given a hidden state
% (o|s).
% Will be normalised in the routine, but useful to define as probabilities
% right away.
%--------------------------------------------------------------------------

a = .75;
b = 1 - a;

A{1}   = [1 1 0 0 0 0 0 0;    % ambiguous starting position (centre)
          0 0 1 1 0 0 0 0;    % safe arm selected and rewarded
          0 0 0 0 a b 0 0;    % risky arm selected and rewarded
          0 0 0 0 b a 0 0;    % risky arm selected and not rewarded
          0 0 0 0 0 0 1 0;    % informative cue - high reward prob
          0 0 0 0 0 0 0 1];   % informative cue - low reward prob
  
clear('a')
clear('b')
 

%% 1.2 Controlled transitions: B{u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% under each action or control state. Here, there are four actions taking the
% agent directly to each of the four locations.
% 
% This is where the Markov property comes in. Transition probabilities only
% depend on the current state and action, not the history of previous
% states or actions.
%--------------------------------------------------------------------------

% move to/stay in the middle
B{1}(:,:,1) = [1 0 0 0 0 0 1 0;  % starting point, high reward context
               0 1 0 0 0 0 0 1;  % starting point, low reward context
               0 0 1 0 0 0 0 0;  % safe option, high reward context
               0 0 0 1 0 0 0 0;  % safe option, low reward context
               0 0 0 0 1 0 0 0;  % risky option, high reward context
               0 0 0 0 0 1 0 0;  % risky option, low reward context
               0 0 0 0 0 0 0 0;  % cue location, high reward context
               0 0 0 0 0 0 0 0]; % cue location, high reward context
           
% move up left to safe  (and check for reward)           
B{1}(:,:,2) = [0 0 0 0 0 0 0 0;  % starting point, high reward context
               0 0 0 0 0 0 0 0;  % starting point, low reward context
               1 0 1 0 0 0 1 0;  % safe option, high reward context
               0 1 0 1 0 0 0 1;  % safe option, low reward context
               0 0 0 0 1 0 0 0;  % risky option, high reward context
               0 0 0 0 0 1 0 0;  % risky option, low reward context
               0 0 0 0 0 0 0 0;  % cue location, high reward context
               0 0 0 0 0 0 0 0]; % cue location, high reward context

% move up right to risky (and check for reward)           
B{1}(:,:,3) = [0 0 0 0 0 0 0 0;  % starting point, high reward context
               0 0 0 0 0 0 0 0;  % starting point, low reward context
               0 0 1 0 0 0 0 0;  % safe option, high reward context
               0 0 0 1 0 0 0 0;  % safe option, low reward context
               1 0 0 0 1 0 1 0;  % risky option, high reward context
               0 1 0 0 0 1 0 1;  % risky option, low reward context
               0 0 0 0 0 0 0 0;  % cue location, high reward context
               0 0 0 0 0 0 0 0]; % cue location, high reward context

% move down (check cue)           
B{1}(:,:,4) = [0 0 0 0 0 0 0 0;  % starting point, high reward context
               0 0 0 0 0 0 0 0;  % starting point, low reward context
               0 0 1 0 0 0 0 0;  % safe option, high reward context
               0 0 0 1 0 0 0 0;  % safe option, low reward context
               0 0 0 0 1 0 0 0;  % risky option, high reward context
               0 0 0 0 0 1 0 0;  % risky option, low reward context
               1 0 0 0 0 0 1 0;  % cue location, high reward context
               0 1 0 0 0 0 0 1]; % cue location, high reward context       

%% 1.3 Priors: 
%==========================================================================

%--------------------------------------------------------------------------
% Now, we have to specify the prior preferences in terms of log
% probabilities. Here, the agent prefers rewarding outcomes
% 
% This is a vector that has the same length as the number of observable
% outcomes (rows of A-matrix). Entries reflect preferences for these
% observations (higher number means higher preferences). These numbers go
% through a softmax in the routine.
%--------------------------------------------------------------------------
cs = 2^1; % preference for safe option
cr = 2^2; % preference for risky option win

% preference for: [staying at starting point | safe | risky + reward | risky + no reward | cue context 1 | cue context 2]
C{1}  = [0 cs cr -cs 0 0]';

%--------------------------------------------------------------------------
% Now specify prior beliefs about initial state
% 
% This is a vector that has the same length as the number of (hidden)
% states (columns of A-matrix), entries reflect beliefs about being 
% in one of these states at the start of the experiment. 
% This vector will be normalised in the routine.
%--------------------------------------------------------------------------
D{1}  = kron([1/2 0 0 0],[1 1])';

%% 1.4 Allowable policies (of depth T).  These are sequences of actions
%==========================================================================

% number of rows    = number of time steps
% number of columns = number of allowed policies
% 
% numbers           = actions
% 1 == go to starting point
% 2 == go to safe option
% 3 == go to risky option
% 4 == go to cue location
V  = [1  1  1  1  2  3  4  4  4  4
      1  2  3  4  2  3  1  2  3  4];
 
 
%% 2. Define MDP Structure
%==========================================================================
%==========================================================================

%--------------------------------------------------------------------------
% Now, just put everything into a big structure (called 'MDP')
%--------------------------------------------------------------------------

mdp.V = V;                    % allowable policies
mdp.A = A;                    % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % prior over initial states
mdp.s = 1;                    % initial state (1 == high reward context, 2 == low reward context)

mdp.eta = 0.5;                % learning rate


%--------------------------------------------------------------------------
% Check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);
 
%--------------------------------------------------------------------------
% Having specified the basic generative model, let's see how active
% infernece solves different tasks
%--------------------------------------------------------------------------


%% 3. Tasks with a random context (i.e., hidden state)
%==========================================================================
%==========================================================================

close all

%% 3.1 Fairly precise behaviour WITH information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(i).s] = deal(2); % randomise initial state

[MDP(1:n).alpha] = deal(16);               % precision of action selection

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1a'); clf
Z_spm_MDP_VB_trial(MDP(1));

%--------------------------------------------------------------------------
% Most relevant neuro output as in plot:
% MDP.X   = Estimates of T states
% MDP.u   = Beliefs about policies
% MDP.P   = Beliefs about actions
% MDP.w   = Posterior expectations of precision (policy)
%--------------------------------------------------------------------------

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

%% 3.2 Fairly precise behaviour WITHOUT information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(i).s] = deal(2); % randomise initial state

[MDP(1:n).alpha] = deal(16);               % precision of action selection

[MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1b'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

%% 3.3 IMprecise behaviour WITH information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(i).s] = deal(2); % randomise initial state

[MDP(1:n).alpha] = deal(2);               % precision of action selection

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1c'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);


%% 4. Fixed context (hidden state), allow for updating of belief about context
%==========================================================================
%==========================================================================

close all

%% 4.1 Fairly precise behaviour WITH information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)]    = deal(MDP); % replicate mdp structure over trials
[MDP(1:n).d]  = deal(mdp.D); % allow for learning of beliefs about hidden states

[MDP(1:n).alpha] = deal(16);               % precision of action selection

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1a'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

%% 4.2 Fairly precise behaviour WITHOUT information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)]    = deal(MDP); % replicate mdp structure over trials
[MDP(1:n).d]  = deal(mdp.D); % allow for learning of beliefs about hidden states

[MDP(1:n).alpha] = deal(16);               % precision of action selection

[MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
% spm_figure('GetWin','Figure 1b'); clf
% Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

%% 4.3 IMprecise behaviour WITHOUT information-gain:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(1:n).d]  = deal(mdp.D); % allow for learning of beliefs about hidden states

[MDP(1:n).alpha] = deal(2);                % precision of action selection

[MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
% spm_figure('GetWin','Figure 1c'); clf
% Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);


%% 5. Simulate lots of trials and look at performance
%==========================================================================
%==========================================================================

close all

%% 5.1 Fairly precise behaviour WITH information-gain and STABLE context (hidden state):
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)]    = deal(MDP); % replicate mdp structure over trials
[MDP(1:n).d]  = deal(mdp.D); % allow for learning of beliefs about hidden states

[MDP(1:n).alpha] = deal(16);               % precision of action selection

N_sims = 20;

Z_MDP_EpistemicLearning_state_lots(MDP,N_sims);

%% 5.2 Fairly precise behaviour WITHOUT information-gain and STABLE context (hidden state):
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)]    = deal(MDP); % replicate mdp structure over trials
[MDP(1:n).d]  = deal(mdp.D); % allow for learning of beliefs about hidden states

[MDP(1:n).alpha] = deal(16);               % precision of action selection

N_sims = 20;

[MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

Z_MDP_EpistemicLearning_state_lots(MDP,N_sims);

%% 5.3 Fairly precise behaviour WITH information-gain and RANDOM context (hidden state):
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(i).s] = deal(2); % randomise initial state

[MDP(1:n).alpha] = deal(16);               % precision of action selection

N_sims = 20;

Z_MDP_EpistemicLearning_state_lots(MDP,N_sims);

%% 5.4 Fairly precise behaviour WITHOUT information-gain and RANDOM context (hidden state):
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials
i         = rand(1,n) > 1/2;  % randomise hidden states over trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials
[MDP(i).s] = deal(2); % randomise initial state

[MDP(1:n).alpha] = deal(16);               % precision of action selection

N_sims = 20;

[MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

Z_MDP_EpistemicLearning_state_lots(MDP,N_sims);

%--------------------------------------------------------------------------
% This completes the exploration of the active inference model. 
% We now turn to the model inversion routines to recover subject specific
% parameters, i.e. 'computational phenotyping'
% 
% This can be done with any data-set from section 3 or 4.
%-------------------------------------------------------------------------