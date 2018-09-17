function Z_MDP_EpistemicLearning_lots(MDP,N_sims)

% Auxiliary function for plotting behaviour in lots of trials

if nargin<2
    N_sims = 100;
end



Prob_risky = [];
Reward     = [];
Trial_num  = [];

if isfield(MDP,'a')
    A_HR = [];
    A_NR = [];
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

for i = 1:size(MDP,2)
    prob_risky(i) = MDP(i).P(3,1);
end

Prob_risky = [Prob_risky;prob_risky'];

for i = 1:size(MDP,2)
    reward(i) = MDP(i).o(2);
end

reward(reward==2) = 1;
reward(reward==4) = 0;
reward(reward==3) = 4;

Reward = [Reward;reward'];

if isfield(MDP,'a')
    
    for i = 1:size(MDP,2)
        a_HR(i) = MDP(i).a{1}(3,3);
        a_NR(i) = MDP(i).a{1}(4,3);
    end

    A_HR = [A_HR; a_HR'];
    A_NR = [A_NR; a_NR'];
end

Trial_num = [Trial_num; [1:n]'];

elapsed_time = toc;

fprintf('### Simulation %d of %d done, time needed: %d ###\n',trial,N_sims,elapsed_time)

end


%% Plot prob to choose risky as a function of concentration params in A:

[unique_A_HR] = unique(A_HR);
[unique_A_NR] = unique(A_NR);

A_risky     = [A_HR A_NR];

[unique_A, ia, ib] = unique(A_risky, 'rows');

prob_risky_unique_A = splitapply(@mean,Prob_risky,ib);

% test = (unique_A+.25)/.5; 
% test(:,2) = max(test(:,2))-(test(:,2)-1); % get y-axis right

A_plot = zeros(length(unique_A_NR),length(unique_A_HR));  % rows: y=noR, colums: x=highR

[~,unique_A_ind_HR] = ismember(unique_A(:,1),unique_A_HR);
[~,unique_A_ind_NR] = ismember(unique_A(:,2),unique_A_NR);

unique_A_ind = [unique_A_ind_HR unique_A_ind_NR];

ind = sub2ind(size(A_plot), max(unique_A_ind(:,2))-unique_A_ind(:,2)+1, unique_A_ind(:,1));
A_plot(ind) = prob_risky_unique_A;

plot_size = min(size(A_plot)); A_plot = A_plot(1:plot_size,1:plot_size);

figure,imagesc(A_plot),title('Probability to choose risky option as a function of observation model'), set(gcf,'color','white')
set(gca, 'YTick', [1:2:plot_size]),set(gca, 'YTickLabel', {[unique_A_NR(plot_size):-1:unique_A_NR(1)]}),ylabel('Certainty no reward')
set(gca, 'XTick', [1:2:plot_size]),set(gca, 'XTickLabel', {[unique_A_HR(1):unique_A_HR(plot_size)]}),xlabel('Certainty high reward')
colorbar,caxis([0 1])


%% Plot prob to choose risky as a function of trials:

prob_risky_trialNum = splitapply(@mean,Prob_risky,Trial_num);

figure,plot(prob_risky_trialNum,'-.','LineWidth',2),title('Probability to choose risky option as a function of time'), set(gcf,'color','white')
set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
ylim([0,1])
set(gca, 'XTick', [0:2:length(prob_risky_trialNum)]),xlabel('Trial Number')

%% Plot cumulative reward as a function of trials:

reward_trialNum = splitapply(@mean,Reward,Trial_num);

figure
plot(cumsum(reward_trialNum),'-.','LineWidth',2)
title('Cumulative reward'), set(gcf,'color','white')
ylim([0,n*4])
ylabel('Reward (Pellets)')
set(gca, 'XTick', [0:2:length(Trial_num)]),xlabel('Trial Number')

end



