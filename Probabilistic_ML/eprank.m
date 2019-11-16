clear all;
close all;

load tennis_data

% ###########################################################################
% ------------------- GET CONVERGENCES ------------------------------------
% sub_range = [1:1:21];
% iters = 10000;     range = [1:1:iters];
% [Ms_container, Ps_container] = MPA(iters, W, G, 0.5);
% figure
% plot(sub_range-1, Ms_container([1, 16, 32, 73, 102], sub_range))
% hold on
% ax = gca;   ax.ColorOrderIndex = 1; %restart the colour order index (so asymptotes are the same colour)
% plot(sub_range-1, ones(1, 21) .* Ms_container([1, 16, 32, 73, 102], iters), '--');
% ylabel("Estimated Mean, \mu");       xlabel("Iteration");
% legend(W([1, 16, 32, 73, 102]))
% hold off
% 
% 
% figure
% plot(sub_range-1, 1./Ps_container([1, 16, 32, 73, 102], sub_range))
% hold on
% ax = gca;   ax.ColorOrderIndex = 1; %restart the colour order index (so asymptotes are the same colour)
% plot(sub_range-1, ones(1, 21) .* (1./Ps_container([1, 16, 32, 73, 102], iters)), '--');
% ylabel("Estimated Variance, \sigma^2");       xlabel("Iteration");
% legend(W([1, 16, 32, 73, 102]))


% ----------------- DIFFERENT INITIALISATIONS -----------------------------
%pv = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100];
% iters = 100;
% params = [0, 1, 2, 5];
% 
% asymp_means = nan(numel(params), numel(params));
% asymp_vars = nan(numel(params), numel(params));
% p1_index = 1;
% for param1 = params
%     p2_index = 1;
%     for param2 = params
%         [Ms_container, Ps_container] = MPA(iters, W, G, param1, param2);
% 
%         asymp_means(p2_index, p1_index) = Ms_container(102, iters); %get the asymptotic value
%         asymp_vars(p2_index, p1_index) = Ps_container(102, iters);
%         p2_index = p2_index+1;
%     end
%     p1_index = p1_index + 1;
% end
% 
% asymp_means
% asymp_vars


% -------------------- Skill and Performance Matrices------------------------------------------
[Ms_container, Ps_container] = MPA(10000, W, G, 0, 0);
Ms_top = Ms_container([16, 1, 5, 11], 10000);
Ps_top = Ps_container([16, 1, 5, 11], 10000);
skill_matrix = nan(4,4);
result_matrix = nan(4,4);
for row = 1:4
    for col = 1:4
        % Find probability one skill is higher than the other
        pmean = Ms_top(row) - Ms_top(col);
        variance = 1/(Ps_top(row)) + 1/(Ps_top(col));
        skill_matrix(row,col) = normcdf((0-pmean)/sqrt(variance));
        t_variance = variance + 1; % + performance inconsistancy
        result_matrix(row,col) = normcdf((0-pmean)/sqrt(t_variance));
    end
end
skill_matrix
result_matrix

% ########################################################################

function [Ms_container, Ps_container] = MPA(totIters, W, G, param1, param2)

    NumPlayers = size(W,1);            % number of players
    NumGames = size(G,1);            % number of games in 2011 season 

    psi = inline('normpdf(x)./normcdf(x)');
    lambda = inline('(normpdf(x)./normcdf(x)).*( (normpdf(x)./normcdf(x)) + x)');
    pv = 0.5;            % prior skill variance (prior mean is always 0)

    % initialize matrices of skill marginals - means and precisions
    Ms = nan(NumPlayers,1);
    Ps = nan(NumPlayers,1);
    % initialize matrices of game to skill messages - means and precisions
    Mgs = param1.*ones(NumGames,2);        Pgs = param2.*ones(NumGames,2); % (originally zeros(NumGames, 2))

    % allocate matrices of skill to game messages - means and precisions
    Msg = nan(NumGames,2);             Psg = nan(NumGames,2);
    
    % initialize Containers for marginal skills
    temp_Ms_container = nan(numel(Ms(:, 1)), totIters); %creates a column for each iter
    temp_Ps_container = nan(numel(Ps(:, 1)), totIters); %rows are player number
    
    %iter = 1;      % max_belief_change = 1;
    for iter = 1:totIters % && (max_belief_change < tol)
      % (1) compute marginal skills 
      for p=1:NumPlayers
        % precision first because it is needed for the mean update
        Ps(p) = 1/pv + sum(Pgs(G==p)); 
        Ms(p) = sum(Pgs(G==p).*Mgs(G==p))./Ps(p);
      end

      % (2) compute skill to game messages
      % precision first because it is needed for the mean update
      Psg = Ps(G) - Pgs;
      Msg = (Ps(G).*Ms(G) - Pgs.*Mgs)./Psg;

      % (3) compute game to performance messages
      vgt = 1 + sum(1./Psg, 2);
      mgt = Msg(:,1) - Msg(:,2); % player 1 always wins the way we store data

      % (4) approximate the marginal on performance differences
      Mt = mgt + sqrt(vgt).*psi(mgt./sqrt(vgt));
      Pt = 1./( vgt.*( 1-lambda(mgt./sqrt(vgt)) ) );

      % (5) compute performance to game messages
      ptg = Pt - 1./vgt;
      mtg = (Mt.*Pt - mgt./vgt)./ptg;   

      % (6) compute game to skills messages
      Pgs = 1./(1 + repmat(1./ptg,1,2) + 1./Psg(:,[2 1]));
      Mgs = [mtg, -mtg] + Msg(:,[2 1]);
      
      temp_Ms_container(:, iter) = Ms;
      temp_Ps_container(:, iter) = Ps;
      %iter = iter+1;
    end
    Ms_container = temp_Ms_container;
    Ps_container = temp_Ps_container;
end

