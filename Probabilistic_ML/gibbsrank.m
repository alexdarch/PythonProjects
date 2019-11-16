close all;
clear all;

load tennis_data    % W = player name e.g. 'Rafael-Nadal'
                    % G = [ [1, 2]; [2, 1]; ...] means 1 played 2 and won,
                    % then in the next game 2 played against 1 and won
                    % i.e. the first entry is the winner and second is the
                    % loser. This is our y_g

randn('seed', 27); % set the pseudo-random number generator seed
NumPlayers = size(W,1);            % number of players
NumGames = size(G,1);            % number of games in 2011 season 

% #########################################################################
% -------------------------- RUN GIBBS SAMPLER ----------------------------
% totITER = 10000;
% w_container = gibbssampler(totITER, G, NumPlayers, NumGames);

% --------------------------- SUB-SAMPLE ----------------------------------
% spacing = 5;
% w_iter = downsample(w_container', spacing)';
% ------------------------- PLOT SKILL VS ITERATION -----------------------
% figure
% plot( [1:totITER], w_container(1:15:70, :) )
% xlabel("Iteration");    ylabel("Player Skill");
% legend([W(1:15:70)])
% xlim([0 1100])
%-------------------------- PLOT AUTO-CORRELATION ------------------------
% plotAutocorr(w_iter)
% --------------------- PLOT MIXING TIME HISTOGRAMS -----------------------
%p = 16;
%plotHistograms(w_iter, p, totITER/spacing)
% --------------------- GELMAN-RUBIN STATISTICS ---------------------------

% range = [10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 2000];
% %range = [10, 20, 50, 100, 200, 1000];
% i = 1;
% chains = 200;d
% r_stats = zeros(numel(range), 3);
% for totIter = range
%     A1 = zeros(chains, totIter);
%     A16 = zeros(chains, totIter);
%     A102 = zeros(chains, totIter);
%     for n = 1:chains
%         w_container = gibbssampler(totIter, G, NumPlayers, NumGames);
%         A1(n, :) = w_container(1, :); %sample for nadal
%         A16(n, :) = w_container(16, :); %sample for djokociv
%         A102(n, :) = w_container(102, :);
%     end
%     r1 = mcmcgr(A1);
%     r16 = mcmcgr(A16);
%     r102 = mcmcgr(A102);
%     r_stats(i, :) = [r1, r16, r102];
%     i = i + 1;
% end

% -------------------------- SKILL TABLES ---------------------------------
totIters = 100000;
w_container = gibbssampler(totIters, G, NumPlayers, NumGames);

% [16, 1, 5, 11]
sW = w_container([16, 1, 5, 11], 1000:5:totIters);          % sample skill
Ms = mean(w_container([16, 1, 5, 11], 1000:5:totIters)'); % Mean skill
Vs = var(w_container([16, 1, 5, 11], 1000:5:totIters)');    % Skill Variance

table_size = 4;

marginal_matrix = nan(table_size, table_size);
joint_matrix = nan(table_size, table_size);
for row = 1:table_size
    for col = 1:table_size
        % Find probability one skill is higher than the other
        pmean = Ms(row) - Ms(col);
        marginal_matrix(row,col) = normcdf((0-pmean)/sqrt(  Vs(row) + Vs(col) ));
        
        % Count samples where skill(p1) > skill(p2). If same add 0.5
        sampled_wins = (sW(col, :) > sW(row, :)) + 0.5*(sW(row, :) == sW(col, :));
        joint_matrix(row,col) = mean(sampled_wins);
    end
end
marginal_matrix
joint_matrix

% ########################################################################





% -------------------------------------------------------------------------
% -------------------------- FUNCTIONS ------------------------------------
function plotHistograms(w, p, totITER)
    %Take the last "converged" samples
    w_end = w(p, totITER-1000:totITER);
    [K_end,X_end] = ksdensity(w_end, 'width', 0.08);
    mu = mean(w_end);     s = sqrt(var(w_end));
    width = 0.1;
    
    figure
     
    subplot(3, 1, 1),
    w1 = w(p, 1:25);
    H_G1 = histogram(w1, "binwidth", width, 'Normalization', 'pdf');  %gives a histogram where the heights are the number of elements per bin
    hold on
    plot(X_end, K_end, 'r-')
    axis([0.5 2.7 0 2.5]);
    plot(0:0.01:2.5 , gauss_distribution(0:0.01:2.5, mu, s), 'b--');    
    title('The First 25 Iterations (N = 1)');
    ylabel('Probability Density');
    xlabel('Player Skill');
    legend("N:N+150 Iterations Histogrammed", "A Kernel Density Estimate of the Last 1000 Iterations", "A Moment-Matched Gaussian with the last 1000 Iterations")
    
    
    subplot(3, 1, 2),
    w2 = w(p, 26:50);
    H_G2 = histogram(w2, "binwidth", width, 'Normalization', 'pdf');  %gives a histogram where the heights are the number of elements per bin
    hold on
    plot(X_end, K_end, 'r-')
    plot(0:0.01:2.5 , gauss_distribution(0:0.01:2.5, mu, s), 'b--');  
    axis([0.5 2.7 0 2.5]);
    title('The Second 25 Iterations (N = 26)');
    ylabel('Probability Density');
    xlabel('Player Skill');
    


    subplot(3, 1, 3),
    w3 = w(p, 51:76);
    H_G3 = histogram(w3, "binwidth", width, 'Normalization', 'pdf');  %gives a histogram where the heights are the number of elements per bin
    hold on
    plot(X_end, K_end, 'r-')
    plot(0:0.01:2.5 , gauss_distribution(0:0.01:2.5, mu, s), 'b--'); 
    axis([0.5 2.7 0 2.5]);
    title('The Third 25 Iterations (N = 51)');
    ylabel('Probability Density');
    xlabel('Player Skill');
    
%     subplot(4, 1, 4),
%     w4 = w(p, 301:400);
%     H_G4 = histogram(w4, 20, 'Normalization', 'pdf');  %gives a histogram where the heights are the number of elements per bin
%     hold on
%     plot(X_end, K_end, 'r-')
%     axis([0.5 2.5 0 2.5]);
%     plot(0:0.01:2.5 , gauss_distribution(0:0.01:2.5, mu, s), 'b--'); 
%     title('The Fourth 100 Iterations');
%     ylabel('Probability');
%     xlabel('Player Skill');
end

function [ R ] = mcmcgr(A)

    [nChains, chainLen] = size(A); % 50rows, totITER columns
    chainMeans = mean(A, 2); % produces a row vector: takes means of columns 
    avgMean = mean(chainMeans);
    var_between_chains = chainLen/(nChains - 1) * sum( (chainMeans - avgMean).^2 );
    
    var_within_chains = sum( var(A.T) ) / nChains;
    
    V = (chainLen-1)*var_within_chains/chainLen + (nChains+1)*var_between_chains/(chainLen*nChains);
    
    
    R = V/var_within_chains;
end

function w_container = gibbssampler(totITER, G, NumPlayers, NumGames)
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
end

function f = gauss_distribution(x, mu, s)
    p1 = -.5 * ((x - mu)/s) .^ 2;
    p2 = (s * sqrt(2*pi));
    f = exp(p1) ./ p2; 
end

function plotAutocorr(w_iter)
    % ------------------ PLOT AUTOCORRELATIONS --------------------
    lag = 50;
    coeff_auto_cov1 = xcov(w_iter(1, :), lag, 'coeff'); %makes the acorr = 1 at 0 lag
    coeff_auto_cov2 = xcov(w_iter(16, :), lag, 'coeff'); %makes the acorr = 1 at 0 lag
    coeff_auto_cov3 = xcov(w_iter(102, :), lag, 'coeff'); %makes the acorr = 1 at 0 lag

    figure
    % Plot the coefficients of the covariance funciton

    plot(linspace( -lag, lag, numel(coeff_auto_cov1) ), coeff_auto_cov1)
    hold on
    plot(linspace( -lag, lag, numel(coeff_auto_cov2) ), coeff_auto_cov2)
    plot(linspace( -lag, lag, numel(coeff_auto_cov3) ), coeff_auto_cov3)
    xlabel("Sample Iteration Lag/Lead");    ylabel("Auto Correlation")
    legend("Rafael-Nadal, id:1", 'Novak-Djokovic, id:16', 'Rohan-Bopanna, id:102')


    figure
    % ----------------- PLOT CORRELATION LENGTH -----------------------
    auto_length1 = cumtrapz(abs(coeff_auto_cov1), linspace( -lag, lag, numel(coeff_auto_cov1)));
    auto_length2 = cumtrapz(abs(coeff_auto_cov2), linspace( -lag, lag, numel(coeff_auto_cov1)));
    auto_length3 = cumtrapz(abs(coeff_auto_cov3), linspace( -lag, lag, numel(coeff_auto_cov1)));
    plot(linspace( -lag, lag, numel(auto_length1)), abs(auto_length1));
    hold on
    plot(linspace( -lag, lag, numel(auto_length2)), abs(auto_length2));
    plot(linspace( -lag, lag, numel(auto_length3)), abs(auto_length3));
    xlabel("Sample Iteration Lag/Lead");       ylabel("Auto Correlation Length");
    legend("Rafael-Nadal, id:1", 'Novak-Djokovic, id:16', 'Rohan-Bopanna, id:102')
end