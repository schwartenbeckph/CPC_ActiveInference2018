function Q = Z_spm_MDP_VB_game_EpistemicLearning(MDP)
% auxiliary plotting routine for spm_MDP_VB - multiple trials
% FORMAT Q = spm_MDP_VB_game(MDP)
%
% MDP.P(M,T)      - probability of emitting action 1,...,M at time 1,...,T
% MDP.Q(N,T)      - an array of conditional (posterior) expectations over
%                   N hidden states and time 1,...,T
% MDP.X           - and Bayesian model averages over policies
% MDP.R           - conditional expectations over policies
% MDP.O(O,T)      - a sparse matrix encoding outcomes at time 1,...,T
% MDP.S(N,T)      - a sparse matrix encoding states at time 1,...,T
% MDP.U(M,T)      - a sparse matrix encoding action at time 1,...,T
% MDP.W(1,T)      - posterior expectations of precision
%
% MDP.un  = un    - simulated neuronal encoding of hidden states
% MDP.xn  = Xn    - simulated neuronal encoding of policies
% MDP.wn  = wn    - simulated neuronal encoding of precision
% MDP.da  = dn    - simulated dopamine responses (deconvolved)
% MDP.rt  = rt    - simulated dopamine responses (deconvolved)
%
% returns summary of performance:
%
%     Q.X  = x    - expected hidden states
%     Q.R  = u    - final policy expectations
%     Q.S  = s    - initial hidden states
%     Q.O  = o    - final outcomes
%     Q.p  = p    - performance
%     Q.q  = q    - reaction times
%
% please see spm_MDP_VB
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_game.m 6763 2016-04-04 09:24:18Z karl $

% numbers of transitions, policies and states
%--------------------------------------------------------------------------
if iscell(MDP(1).X)
    Nf = numel(MDP(1).B);                 % number of hidden state factors
    Ng = numel(MDP(1).A);                 % number of outcome factors
else
    Nf = 1;
    Ng = 1;
end

% graphics
%==========================================================================
Nt    = length(MDP);               % number of trials
Ne    = size(MDP(1).V,1) + 1;      % number of epochs per trial
Np    = size(MDP(1).V,2) + 1;      % number of policies
for i = 1:Nt
    
    % assemble expectations of hidden states and outcomes
    %----------------------------------------------------------------------
    for j = 1:Ne
        for k = 1:Ne
            for f = 1:Nf
                try
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn{f}(:,:,j,k)')';
                catch
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn(:,:,j,k)')';
                end
            end
        end
    end
    s(:,i) = MDP(i).s(:,1);
    o(:,i) = MDP(i).o(:,end);
    u(:,i) = MDP(i).R(:,end);
    w(:,i) = mean(MDP(i).dn,2);
    
    
    % assemble context learning
    %----------------------------------------------------------------------
    for f = 1:Nf
        try
            try
                D = MDP(i).d{f};
            catch
                D = MDP(i).D{f};
            end
        catch
            try
                D = MDP(i).d;
            catch
                D = MDP(i).D;
            end
        end
        d{f}(:,i) = D/sum(D);
    end
    
    % assemble performance
    %----------------------------------------------------------------------
    p(i)  = 0;
    for g = 1:Ng
        try
            U = spm_softmax(MDP(i).C{g});
        catch
            U = spm_softmax(MDP(i).C);
        end
        for t = 1:Ne
            p(i) = p(i) + log(U(MDP(i).o(g,t),t))/Ne;
        end
    end
    q(i)   = sum(MDP(i).rt(2:end));
    
end

% assemble output structure if required
%--------------------------------------------------------------------------
if nargout
    Q.X  = x;            % expected hidden states
    Q.R  = u;            % final policy expectations
    Q.S  = s;            % inital hidden states
    Q.O  = o;            % final outcomes
    Q.p  = p;            % performance
    Q.q  = q;            % reaction times
    return
end

%% Now plot stuff

if isfield(MDP,'a')
    n_rows = 5; % number of rows in subplot
else
    n_rows = 2;
end

% Initial states and expected policies (habit in red)
%--------------------------------------------------------------------------
choice_prob = [];
for i=1:Nt, choice_prob = [choice_prob MDP(i).P]; end

col   = {'b.','m.','g.','r.','c.','k.'};
cols = [0:1/32:1; 0:1/32:1; 0:1/32:1]';
t     = 1:Nt;
figure, set(gcf,'color','white')

subplot(n_rows,1,1)

if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end

im = imagesc((1 - choice_prob)); colormap(cols),hold on


choice_conflict = sum(choice_prob.*log(choice_prob));

chosen_action = [1 2 3]*u; plot(chosen_action,col{1},'MarkerSize',MarkerSize)

try
    plot(Np*(1 - u(Np,:)),'r')
end
try
    E = spm_softmax(spm_cat({MDP.e}));
    plot(Np*(1 - E(end,:)),'r:')
end
title('Initial state and policy selection')
set(gca, 'XTick', [0:Nt]), set(gca, 'YTickLabel', {'Stay','Safe','Risky'})

xlabel('Trial'),ylabel('Policy')
hold off


% Performance
%--------------------------------------------------------------------------
q     = q - mean(q);
q     = q/std(q);

subplot(n_rows,1,2), bar(p,'k'),   hold on % utility of outcome

plot(choice_conflict,'.c','MarkerSize',16), hold on % simulated reaction time
plot(choice_conflict,':c')

% o(1,:) = o(1,:)+3; % Avoid same colours for different factors
for g = 1:Ng
    for i = 1:max(o(g,:))             % first row of o: risky (3) or safe (2) outcome, second row: reward (3=high r, 3=no r, 2=low r)
        j = find(o(g,:) == i);
        plot(t(j),j - j + 3 + g,col{rem(i - 1,6)+ 1},'MarkerSize',MarkerSize)
    end
end
title('Final outcome, performance and choice conflict')
xlabel('Trial'),ylabel('Utility Outcomes'), spm_axis tight, set(gca, 'XTick', [0:Nt]), %xticks(0:Nt);
hold off

if isfield(MDP,'a')
    cols = [0:1/32:1; 0:1/32:1; 0:1/32:1]';

    n = size(MDP,2);
    a_fig = MDP(1).a{1};
    
    for i = 0:n
        if i == 0
            subplot(n_rows,11,23 + i)
            imagesc(1-a_fig), colormap(cols), title(sprintf('Start'))
            set(gca, 'YTick', [1:4]), set(gca, 'YTickLabel', {'SP','LR','HR','NR'}), set(gca, 'XTickLabel', {'SP','S','R'})
            axis image
        else
            subplot(n_rows,11,23 + i)

            a_fig(3,3) = MDP(i).a{1}(3,3); a_fig(4,3) = MDP(i).a{1}(4,3); 

            if MDP(i).s(2)==3 % for plotting purposes 
                a_fig(1,1) = MDP(1).a{1}(1,1)+i*MDP(1).eta; 
                a_fig(2,2) = MDP(1).a{1}(2,2)+i*MDP(1).eta;
            end 

            a_fig_use = a_fig/diag(sum(a_fig));

            imagesc(1-a_fig_use), colormap(cols), title(sprintf('trial %i',i))
            set(gca, 'YTick', [1:4]), set(gca, 'YTickLabel', {'SP','LR','HR','NR'}), set(gca, 'XTickLabel', {'SP','S','R'})
            axis image
        end
    end

end
