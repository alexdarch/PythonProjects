close all;
clear all;

load tennis_data    
randn('seed', 27); % set the pseudo-random number generator seed

NumPlayers = size(W,1);            % number of players
NumGames = size(G,1);            % number of games in 2011 season 

gibbsIters = 100000;     msgIters = 1000;
[Ms_container, Ps_container] = MPA(msgIters, W, G, 0, 0);
[w_container, iSS_final, mu_final] = gibbssampler(gibbsIters, G, NumPlayers, NumGames);

% ------------------------ EMPERICAL GAME WINS ----------------------------
wins = nan(NumPlayers, 1);
games = nan(NumPlayers, 1);
 for p = 1:NumPlayers
    wins(p) = sum(   ( (G(:,1) - p) == 0 )   ); 
    games(p) = sum(   ( (G(:,1) - p) == 0 )  +  ( (G(:, 2) - p) == 0 )   );
 end
frac_wins = wins./games;
empericalRanking = normalize(frac_wins, 'range', [0, 1]);
barChart(empericalRanking, W);

% ------------------------ GIBBS PREDICTIONS ------------------------------
w_container(:, gibbsIters);
gibbsMean = mean(w_container(:, 1000:5:gibbsIters), 2);
gibbsRanking = normalize(gibbsMean, 'range', [0, 1]);
barChart(gibbsRanking, W);

% ------------------------ MESSAGE PREDICTIONS ----------------------------

epMean = Ms_container(:, msgIters);
epRanking = normalize(epMean, 'range', [0, 1]);
barChart(epRanking, W);


function barChart(P, W)
    % make a bar plot from vector P and annotate with player names from W
    figure;
    [kk, ii] = sort(P, 'descend');

    np = 107;
    kk(np:-1:1)
    yyaxis left
    barh(kk(np:-1:1))
    set(gca, 'YTickLabel', W(ii(np:-1:1)), 'YTick', 1:np, 'FontSize', 8)
    axis([0 1 0.5 np+0.5])
    xlabel("Normalised Player Skill")
    
    % Plot skill-density
    hold on;
    %[skillDensity, xDensity] = ksdensity(kk(np:-1:1), 'width', 0.2);
    yyaxis right
    ylim([0, 3.8])
    histogram(kk(np:-1:1), 'binwidth', 0.05, 'Normalization', 'pdf')
    ylabel("Skill Probability Density ")
    %plot(xDensity, skillDensity, 'r-');
    %mu = mean(w_end);     s = sqrt(var(w_end));
    %axis([0.5 2.7 0 2.5]);
    %plot(0:0.01:2.5 , gauss_distribution(0:0.01:2.5, mu, s), 'b--');  
end

function [w_container, iSS_final, mu_final] = gibbssampler(totITER, G, NumPlayers, NumGames)
    w_container = zeros(NumPlayers, totITER);
    pv = 0.5*ones(NumPlayers,1);           % prior skill variance 
    w = zeros(NumPlayers,1);               % set skills to prior mean
    
    for i = 1:totITER

      % First, sample performance differences given the skills and outcomes
      t = nan(NumGames,1); % contains a t_g variable for each game
      for g = 1:NumGames   % loop over games
        s = w(G(g,1))-w(G(g,2));  % difference in skills e.g. w(G(g, 1)) = skill of player G(g, 1), where g is the gth game played. 
                                  % e.g. in the 50th game, player 19 played player 22. the skill difference to start is
                                  % 0 because we initialised w = zeros(numPlayers, 1)
        t(g) = randn()+s;         % performace difference sample
        while t(g) < 0  % rejection sampling: only positive perf diffs accepted
          t(g) = randn()+s; % if rejected, sample again
        end
      end 

      % Second, jointly sample skills given the performance differences 
      m = nan(NumPlayers,1); 
      for p = 1:NumPlayers
        % ((G(:,1) - p) == 0) generates a vector of 1's, where G(:, 1) is the
        % vector of % players that won)and p is the current player id.
        m(p) = t'*(    ( (G(:,1) - p) == 0 )  -  ( (G(:, 2) - p) == 0 )    );                                                         
      end

      iS = zeros(NumPlayers, NumPlayers);                                 
      for g = 1:NumGames
          iS(G(g,1), G(g,1)) = iS(G(g,1), G(g,1)) + 1;
          iS(G(g,2), G(g,2)) = iS(G(g,2), G(g,2)) + 1;
          iS(G(g,1), G(g,2)) = iS(G(g,1), G(g,2)) - 1;
          iS(G(g,2), G(g,1)) = iS(G(g,2), G(g,1)) - 1;
      end

      iSS = diag(1./pv) + iS; % posterior precision matrix
      % prepare to sample from a multivariate Gaussian
      % Note: inv(M)*z = R\(R'\z) where R = chol(M);
      iR = chol(iSS);  % Cholesky decomposition of the posterior precision matrix
      mu = iR\(iR'\m); % equivalent to inv(iSS)*m but more efficient

      % sample from N(mu, inv(iSS))
      w = mu + iR\randn(NumPlayers,1);

      w_container(:, i) = w; %a container of all the player skills over time
      if (mod(i, totITER/100) == 0)
          fprintf("%i%%\t", 100*i/totITER)
      end
    end
    fprintf("\n")
    iSS_final = iSS;
    mu_final = mu;
end

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