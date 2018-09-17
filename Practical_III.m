% Practical for Comp Psych in Zurich 2018
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part III %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

% Before you start, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder, where the practicals live, to your path in Matlab.

%% 12. Set up model structure
%==========================================================================
%==========================================================================

rng('shuffle')

%% 12.1 Outcome probabilities: A
%==========================================================================

a = 0.5;
b = 1 - a;

% Location and Reward, exteroceptive - no uncertainty about location, interooceptive - uncertainty about reward prob
%--------------------------------------------------------------------------
% That's were true reward prob comes in
A{1} = [1 0 0          % reward neutral  (starting position)
        0 1 0          % low reward      (safe option)
        0 0 a          % high reward     (risky option)
        0 0 b];        % negative reward (risky option)
    
clear('a'),clear('b')    
    
%% 12.2 Beliefs about outcome (likelihood) mapping
%==========================================================================

%--------------------------------------------------------------------------
% That's where learning comes in - start with uniform prior
%--------------------------------------------------------------------------

a{1} = [1 0 0          % reward neutral  (starting position)
        0 1 0          % low reward      (safe option)
        0 0 1/4        % high reward     (risky option)
        0 0 1/4];      % negative reward (risky option)
    
%% 12.3 Controlled transitions: B{u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% for each factor. Here, there are three actions taking the agent directly
% to each of the three locations.
%--------------------------------------------------------------------------
B{1}(:,:,1) = [1 1 1; 0 0 0;0 0 0];     % move to the starting point
B{1}(:,:,2) = [0 0 0; 1 1 1;0 0 0];     % move to safe  option (and check for reward)
B{1}(:,:,3) = [0 0 0; 0 0 0;1 1 1];     % move to risky option (and check for reward)

%% 12.4 Priors: 
%==========================================================================

%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities over outcomes. Here, the agent prefers high rewards over
% low rewards over no rewards
%--------------------------------------------------------------------------

cs = 2^1; % preference for safe option
cr = 2^2; % preference for risky option win

C{1}  = [0 cs cr -cs]'; % preference for: [staying at starting point | safe | risky + reward | risky + no reward]

%--------------------------------------------------------------------------
% Now specify prior beliefs about initial state
%--------------------------------------------------------------------------
D{1}  = [1 0 0]'; % prior over starting point - rat 'starts' at starting point (not at safe or risky option)


%% 13.5 Allowable policies (of depth T).  These are sequences of actions
%==========================================================================

V     = [1 2 3]; % stay, go left, go right

%% 14. Define MDP Structure
%==========================================================================
%==========================================================================

mdp.V = V;                    % allowable policies
mdp.A = A;                    % observation process
mdp.a = a;                    % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % prior over initial states
mdp.s = 1;

mdp.eta   = 0.5;              % Learning rate

%--------------------------------------------------------------------------
% Check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);
 
%--------------------------------------------------------------------------
% Having specified the basic generative model, let's see how active
% infernece solves different tasks
%--------------------------------------------------------------------------

%% 15. Simulate behaviour
%==========================================================================
%==========================================================================

%% 15.1 Fairly precise behaviour WITHOUT active learning:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP);  % replicate mdp structure over trials

[MDP(1:n).alpha] = deal(4);                % precision of action selection

[MDP(1:n).curiosity] = deal(false);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1a'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning(MDP);

%% 15.2 Fairly precise behaviour WITH active learning:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials

[MDP(1:n).alpha] = deal(4);               % precision of action selection

[MDP(1:n).curiosity] = deal(true);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1b'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning(MDP);
 
%% 15.3 Fairly IMprecise behaviour WITHOUT active learning:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials

[MDP(1:n).alpha] = deal(1);                % precision of action selection

[MDP(1:n).curiosity] = deal(false);        % sensitivity for epistemic value

MDP  = Z_spm_MDP_VB_X(MDP);

% illustrate behavioural responses – single trial
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1c'); clf
Z_spm_MDP_VB_trial(MDP(1));

% illustrate behavioural responses over trials
%--------------------------------------------------------------------------
Z_spm_MDP_VB_game_EpistemicLearning(MDP);

%% 16. Simulate lots of trials and look at performance
%==========================================================================
%==========================================================================

close all

%% 16.1 Fairly precise behaviour WITH active learning:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials

[MDP(1:n).alpha] = deal(4);                % precision of action selection

[MDP(1:n).curiosity] = deal(true);        % sensitivity for epistemic value

N_sims = 100;

Z_MDP_EpistemicLearning_lots(MDP,N_sims);

%% 16.2 Fairly precise behaviour WITHOUT active learning:
%==========================================================================

% number of simulated trials
%--------------------------------------------------------------------------
n         = 32;               % number of trials

MDP       = mdp;
 
[MDP(1:n)] = deal(MDP); % replicate mdp structure over trials

[MDP(1:n).alpha] = deal(4);                % precision of action selection

[MDP(1:n).curiosity] = deal(false);        % sensitivity for epistemic value

N_sims = 100;

Z_MDP_EpistemicLearning_lots(MDP,N_sims);

