\# Creating an Index Fund Using AI-Driven Decision-Making Approaches

\## Slide 1: Title Slide - \*\*Title:\*\* Creating an Index Fund Using
AI-Driven Decision-Making Approaches - \*\*Subtitle:\*\* Tracking the
S&P 100 with Optimization and Clustering - \*\*Team Members:\*\* \[Your
Name\] and \[Your Teammate\'s Name\] - \*\*Course:\*\* H9AIDM: AI Driven
Decision Making

\## Slide 2: Project Overview - \*\*Objective:\*\* Create an index fund
tracking the S&P 100 using fewer than 100 stocks - \*\*Parameter q:\*\*
Number of stocks to select (5-30) - \*\*Performance Goal:\*\* Track S&P
100 performance across multiple time horizons - \*\*Metrics:\*\*
Correlation, tracking error, R-squared - \*\*Approaches:\*\* Two
AI-driven methodologies

\## Slide 3: Why This Matters - \*\*Cost Reduction:\*\* Lower
transaction costs and management fees - \*\*Operational Efficiency:\*\*
Simpler portfolio management - \*\*Improved Liquidity:\*\* Focus on more
liquid securities - \*\*Customization Potential:\*\* Easier to modify
for specific needs - \*\*Real-World Applications:\*\* ETFs, personalized
indexing, factor tilting

\## Slide 4: Approach 1 - Optimization with AMPL - \*\*Description:\*\*
Mathematical optimization using AMPL + CPLEX - \*\*Objective
Function:\*\* Maximize correlation with benchmark while minimizing
risk - \*\*Key Constraints:\*\*  - Exactly q stocks selected  - Weights
sum to 1  - Maximum weight per stock (10%) - \*\*Pipeline:\*\* Data →
AMPL Model → Solver → Selected Stocks + Weights

\## Slide 5: AMPL Model \`\`\` \# Decision Variables var Select{i in
TICKERS} binary; var Weight{i in TICKERS} \>= 0, \<= max_weight;

\# Objective maximize Portfolio_Objective: correlation_weight \* sum{i
in TICKERS} benchmark_corr\[i\] \* Weight\[i\] - risk_weight \* sum{i in
TICKERS, j in TICKERS} Weight\[i\] \* Weight\[j\] \* covariance\[i,j\];

\# Constraints subject to Total_Stocks: sum{i in TICKERS} Select\[i\] =
q; subject to Total_Weight: sum{i in TICKERS} Weight\[i\] = 1; subject
to Weight_If_Selected{i in TICKERS}: Weight\[i\] \<= Select\[i\] \*
max_weight; \`\`\`

\## Slide 6: Approach 2 - Clustering - \*\*Description:\*\* K-means
clustering of stocks based on financial features - \*\*Features
Used:\*\*  - Correlation with benchmark  - Volatility (standard
deviation)  - Average return  - Beta coefficient - \*\*Process:\*\*
Group similar stocks → Select representatives from each cluster -
\*\*Weight Determination:\*\* Initially equal, later price-weighted

\## Slide 7: Clustering Visualization - \*\*Image:\*\* PCA visualization
of clusters (from outputs) - \*\*Key Elements:\*\*  - Each point
represents a stock  - Colors indicate cluster membership  - Circled
points represent selected stocks  - Axes are principal components of
financial features

\## Slide 8: Results - Correlation - \*\*Chart:\*\* Bar chart comparing
correlation across q values (5-30) - \*\*Key Findings:\*\*  -
Optimization achieves higher correlation (up to 0.956)  - Correlation
improves with increasing q  - Both approaches achieve \>0.9 correlation
at q=30  - Optimization fails at q=5

\## Slide 9: Results - Tracking Error - \*\*Chart:\*\* Bar chart
comparing tracking error across q values - \*\*Key Findings:\*\*  -
Clustering consistently achieves lower tracking error  - Optimization
has 2-4x higher tracking error  - Both approaches show improving trends
with higher q  - Best tracking error: 0.065 (Clustering, q=30)

\## Slide 10: Portfolio Composition - \*\*Selected Stocks:\*\* Visual
comparison of stocks selected by each approach - \*\*Weight
Distribution:\*\*  - Optimization: Concentrated weights at maximum (10%)
 - Clustering: More evenly distributed weights - \*\*Sector
Diversity:\*\*  - Optimization: Tech and financial heavy  - Clustering:
Better sector representation

\## Slide 11: Time Horizon Analysis - \*\*Chart:\*\* Line graph of
correlation/tracking error across time horizons - \*\*Key Insights:\*\*
 - Performance consistent across time horizons  - Short-term (3-month)
tracking slightly more challenging  - Both approaches maintain relative
advantages  - Longer horizons show marginally better performance

\## Slide 12: Key Findings - \*\*Trade-offs:\*\* Higher correlation
(optimization) vs. lower tracking error (clustering) - \*\*Stock
Selection:\*\* Only 30% overlap between approaches - \*\*Impact of
q:\*\* Diminishing returns beyond q=20 - \*\*Portfolio Construction:\*\*
 - Optimization tends toward concentration  - Clustering provides
natural diversification - \*\*Best Configuration:\*\* q=30 for both
approaches

\## Slide 13: Challenges & Limitations - \*\*Data Challenges:\*\*
Missing values, limited history - \*\*Computational Issues:\*\*
Optimization convergence problems - \*\*Parameter Sensitivity:\*\*
Results affected by constraint choices - \*\*Market Changes:\*\*
Historical data may not predict future - \*\*Feature Selection:\*\*
Limited feature set for clustering

\## Slide 14: Future Improvements - \*\*Hybrid Approach:\*\* Combine
clustering + optimization - \*\*Enhanced Features:\*\* Add sector data,
fundamentals - \*\*Dynamic Rebalancing:\*\* Test periodic portfolio
updates - \*\*Alternative Algorithms:\*\* Test hierarchical clustering,
genetic algorithms - \*\*Transaction Costs:\*\* Incorporate cost models

\## Slide 15: Conclusion - \*\*Both approaches successfully track S&P
100 with fewer stocks\*\* - \*\*Optimization maximizes correlation,
clustering minimizes tracking error\*\* - \*\*Best performance achieved
at q=30 stocks (30% of index)\*\* - \*\*Different selection strategies
suit different investor priorities\*\* - \*\*AI-driven approaches offer
powerful tools for index fund construction\*\*

\## Slide 16: Q&A - Thank You - Questions? - Contact Information
