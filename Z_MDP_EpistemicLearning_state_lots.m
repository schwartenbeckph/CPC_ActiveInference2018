function Z_MDP_EpistemicLearning_state_lots(MDP,N_sims)


% Auxiliary function for plotting behaviour in lots of trials

if nargin<2
    N_sims = 100;
end

Prob_cue  = [];
Trial_num = [];
Reward    = [];

if isfield(MDP,'d')
    D_HR = [];
    D_LR = [];
end

n   = size(MDP,2);  % number of trials
mdp = MDP;


for trial = 1:N_sims

elapsed_time = 0;
tic
    
MDP        = mdp;

% solve and show behaviour over trials (and epistemic learning)
%--------------------------------------------------------------------------
MDP  = Z_spm_MDP_VB_X(MDP);

choice_prob = extractfield(MDP, 'P'); choice_prob = reshape(choice_prob,4,length(choice_prob)/4); choice_prob(:,2:2:n*2) = [];
prob_cue = choice_prob(4,:);

Prob_cue = [Prob_cue;prob_cue'];

if isfield(MDP,'d')
    d = extractfield(MDP, 'd');
    d_HR = zeros(1,n);
    d_LR = zeros(1,n);

    for i = 1:n
        d_HR(i) = d{i}{1}(1);
        d_LR(i) = d{i}{1}(2);
    end

    D_HR = [D_HR; d_HR'];
    D_LR = [D_LR; d_LR'];

end

% Obtained reward - 1 pellet in safe option, 0/4 pellets in risky option
reward      = extractfield(MDP, 'o'); reward = reshape(reward,3,length(reward)/3); 
reward      = reward(3,:);
reward(reward==1) = 0;
reward(reward==5) = 0;
reward(reward==6) = 0;
reward(reward==2) = 1;
reward(reward==4) = 0;
reward(reward==3) = 4;

Reward = [Reward;reward'];

Trial_num = [Trial_num; [1:n]'];

elapsed_time = toc;

fprintf('### Simulation %d of %d done, time needed: %d ###\n',trial,N_sims,elapsed_time)

end



%% Plot prob to choose cue

if isfield(MDP,'d')
    
    D_cue     = D_HR./sum([D_HR,D_LR],2);
    D_cue     = (D_cue.*log(D_cue) + (1-D_cue).*log(1-D_cue));

    [unique_D, ~, ib] = unique(D_cue, 'rows');

    prob_cue_unique_D = splitapply(@mean,Prob_cue,ib);

    figure,plot(unique_D,prob_cue_unique_D,'-.','LineWidth',2)
    title(sprintf('Probability to sample the cue first as a function of certainty, %d simulations',N_sims)), set(gcf,'color','white')
    set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
    ylim([0,1.2])
    xlabel('Certainty (entropy) about Context')

end


%% Plot prob to choose cue as a function of trials:

prob_cue_trialNum = splitapply(@mean,Prob_cue,Trial_num);

figure,plot(prob_cue_trialNum,'-.','LineWidth',2)
title(sprintf('Probability to sample the cue first as a function of time, %d simulations',N_sims)), set(gcf,'color','white')
set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
ylim([0,1.2])
set(gca, 'XTick', [0:2:length(prob_cue_trialNum)]),xlabel('Trial Number')


%% Plot cumulative reward as a function of trials:

reward_trialNum = splitapply(@mean,Reward,Trial_num);

figure
plot(cumsum(reward_trialNum),'-.','LineWidth',2)
title('Cumulative reward'), set(gcf,'color','white')
ylim([0,n*4])
ylabel('Reward (Pellets)')
set(gca, 'XTick', [0:2:length(Trial_num)]),xlabel('Trial Number')

end


