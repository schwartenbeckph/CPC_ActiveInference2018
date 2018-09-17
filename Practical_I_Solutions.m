% Solutions for exercises on task I Comp Psych in Zurich 2018

clear
close all

% Before you start, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder, where the practicals live, to your path in Matlab.

task = 1; % specify here which task you want to solve of {1,2,3,4,5,6}
% task 1 illustrates that 'hidden state exploration' only makes sense if
% the cue is informative

% task 2 illustrates that information gain and reward are additive

% task 3 illustrates that information gain depends on prior uncertainty

% task 4 shows that you can use exactly this setup to model reversal
% learning tasks

% task 5 shows that the learning rate determines how quickly you switch to
% exploitation, but only if there is a stable context

% task 6 shows that it can be adaptive to be random if you are insensitive
% to information gain
 
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
      
if task == 1
    A{1}   = [1 1 0 0 0 0 0   0;      % ambiguous starting position (centre)
              0 0 1 1 0 0 0   0;      % safe arm selected and rewarded
              0 0 0 0 a b 0   0;      % risky arm selected and rewarded
              0 0 0 0 b a 0   0;      % risky arm selected and not rewarded
              0 0 0 0 0 0 0.5 0.5;    % uninformative cue - high reward prob
              0 0 0 0 0 0 0.5 0.5];   % uninformative cue - low reward prob
else
    A{1}   = [1 1 0 0 0 0 0 0;    % ambiguous starting position (centre)
              0 0 1 1 0 0 0 0;    % safe arm selected and rewarded
              0 0 0 0 a b 0 0;    % risky arm selected and rewarded
              0 0 0 0 b a 0 0;    % risky arm selected and not rewarded
              0 0 0 0 0 0 1 0;    % informative cue - high reward prob
              0 0 0 0 0 0 0 1];   % informative cue - low reward prob
end

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
if task == 2
    C{1}  = [0 0 0 0 0 0]'; % uniform preferences
    % C{1}  = [0 cs cs -cs 0 0]'; % equal preferences for safe and risky
    % C{1}  = [0 cs cr 0 0 0]'; % agent doesn't mind to lose
else
    C{1}  = [0 cs cr -cs 0 0]';
end
%--------------------------------------------------------------------------
% Now specify prior beliefs about initial state
% 
% This is a vector that has the same length as the number of (hidden)
% states (columns of A-matrix), entries reflect beliefs about being 
% in one of these states at the start of the experiment. 
% This vector will be normalised in the routine.
%--------------------------------------------------------------------------
if task == 3
    D{1}  = [0.99 0.01 0 0 0 0 0 0 ]';  % certain belief that reward statistics is high
    % D{1}  = [0.01 0.99 0 0 0 0 0 0 ]'; % certain belief that reward statistics is low
else
    D{1}  = kron([1/4 0 0 0],[1 1])';
end

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
mdp.s = 1;                    % initial state

if task == 5
    mdp.eta = 0.3;                % low learning rate
%     mdp.eta = 1;                % high learning rate
else
    mdp.eta = 0.5;                % learning rate
end
%--------------------------------------------------------------------------
% Check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);
 
%--------------------------------------------------------------------------
% Having specified the basic generative model, let's see how active
% infernece solves different tasks
%--------------------------------------------------------------------------

close all

if task == 4
    
    %% Fairly precise behaviour WITH information-gain, reversal learning:
    %==========================================================================

    % number of simulated trials
    %--------------------------------------------------------------------------
    n         = 32;               % number of trials

    MDP       = mdp;

    [MDP(1:n)]    = deal(MDP);
    [MDP(1:n).d]  = deal(mdp.D);

    [MDP(n/2+1:n).s] = deal(2); % reversal learning
    
%     [MDP(1:n/2).s] = deal(2);
%     [MDP(n/2+1:n).s] = deal(1);
    
    [MDP(1:n).alpha] = deal(16);               % precision of action selection

    MDP  = Z_spm_MDP_VB_X(MDP);

    % illustrate behavioural responses – single trial
    %--------------------------------------------------------------------------
    % spm_figure('GetWin','Figure 1a'); clf
    % Z_spm_MDP_VB_trial(MDP(1));

    % illustrate behavioural responses over trials
    %--------------------------------------------------------------------------
    Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

elseif task == 6
    
    %% Imprecise behaviour WITHOUT information-gain
    %==========================================================================

    % number of simulated trials
    %--------------------------------------------------------------------------
    n         = 32;               % number of trials

    MDP       = mdp;

    [MDP(1:n)] = deal(MDP);
    [MDP(1:n).d]  = deal(mdp.D);

    [MDP(1:n).alpha] = deal(6);                % low precision of action selection

    [MDP(1:n).ambiguity] = deal(false);        % sensitivity for epistemic value

    MDP  = Z_spm_MDP_VB_X(MDP);

    % illustrate behavioural responses – single trial
    %--------------------------------------------------------------------------
    % spm_figure('GetWin','Figure 1a'); clf
    % Z_spm_MDP_VB_trial(MDP(1));

    % illustrate behavioural responses over trials
    %--------------------------------------------------------------------------
    Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

else   
    
    %% Fairly precise behaviour WITH information-gain, random context:
    %==========================================================================

    % number of simulated trials
    %--------------------------------------------------------------------------
    n         = 32;               % number of trials
    i         = rand(1,n) > 1/2;  % randomise hidden states over trials

    MDP       = mdp;

    [MDP(1:n)] = deal(MDP);
    [MDP(i).s] = deal(2);

    [MDP(1:n).alpha] = deal(16);               % precision of action selection

    MDP  = Z_spm_MDP_VB_X(MDP);

    % illustrate behavioural responses – single trial
    %--------------------------------------------------------------------------
    % spm_figure('GetWin','Figure 1a'); clf
    % Z_spm_MDP_VB_trial(MDP(1));

    % illustrate behavioural responses over trials
    %--------------------------------------------------------------------------
    Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);

    %% Fairly precise behaviour WITH information-gain, stable context::
    %==========================================================================

    % number of simulated trials
    %--------------------------------------------------------------------------
    n         = 32;               % number of trials

    MDP       = mdp;

    [MDP(1:n)]    = deal(MDP);
    [MDP(1:n).d]  = deal(mdp.D);

    [MDP(1:n).alpha] = deal(16);               % precision of action selection

    MDP  = Z_spm_MDP_VB_X(MDP);

    % illustrate behavioural responses – single trial
    %--------------------------------------------------------------------------
    % spm_figure('GetWin','Figure 1a'); clf
    % Z_spm_MDP_VB_trial(MDP(1));

    % illustrate behavioural responses over trials
    %--------------------------------------------------------------------------
    Z_spm_MDP_VB_game_EpistemicLearning_state(MDP);
    
end

