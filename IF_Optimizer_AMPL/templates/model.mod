
    # Sets and Parameters
    param n;                                  # Number of available stocks
    param q;                                  # Number of stocks to select
    param max_weight;                         # Maximum weight for any single stock
    param correlation_weight;                 # Weight for correlation component
    param risk_weight;                        # Weight for risk component
    
    set TICKERS;                              # Set of stock tickers
    
    param expected_return{i in TICKERS};      # Expected return for each stock
    param benchmark_corr{i in TICKERS};       # Correlation with benchmark
    param covariance{i in TICKERS, j in TICKERS}; # Covariance matrix
    
    # Decision Variables
    var Select{i in TICKERS} binary;          # 1 if stock i is selected, 0 otherwise
    var Weight{i in TICKERS} >= 0, <= max_weight; # Weight of stock i in portfolio (with upper bound)
    
    # Objective: Balance between maximizing correlation and minimizing risk
    maximize Portfolio_Objective:
        correlation_weight * sum{i in TICKERS} benchmark_corr[i] * Weight[i] - 
        risk_weight * sum{i in TICKERS, j in TICKERS} Weight[i] * Weight[j] * covariance[i,j];
    
    # Constraints
    subject to Total_Stocks:
        sum{i in TICKERS} Select[i] = q;
    
    subject to Total_Weight:
        sum{i in TICKERS} Weight[i] = 1;
    
    subject to Weight_If_Selected{i in TICKERS}:
        Weight[i] <= Select[i] * max_weight;
        
    # Minimum weight constraint (if selected)
    subject to Min_Weight_If_Selected{i in TICKERS}:
        Weight[i] >= Select[i] * 0.001;
    
    # Sector diversification could be added here if sector data was available
    