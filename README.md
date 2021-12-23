# Style Rotation on Swiss Long-Only Equity Factors

Project on "Style Rotation on Swiss Long-Only Equity Factors" as part of the course Quantitative Asset & Risk Management II - HEC Lausanne - Autumn 2021

![alt text](https://camo.githubusercontent.com/c327657381291ed9f2e8866cb96ac4861431d9c244b7b14dcf4e1470cbf632da/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f612f61332f4845435f4c617573616e6e655f6c6f676f2e7376672f32393370782d4845435f4c617573616e6e655f6c6f676f2e7376672e706e67)

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

Supervisor: Prof. Fabio Alessandrini

## Objective
* Our client is a large Swiss pension fund who has a substantial allocation to Swiss equities. He is a firm believer of risk premia and is fully convinced by the long-term benefits of tilting his portfolio to reap the benefits of well-known risk premia.
* With no particular view on which risk premia is best suited for him, he wants to go for a diversified approach. He is nevertheless concerned by the time-varying nature of factor returns and fears of being unable to cope with a too long period of underperformance of one given factor.
* He is therefore thinking about the potentials of adjusting his exposure to the various risk premia over time and make his portfolio more dynamic.
* He is willing to give a mandate for managing a dynamic long-only portfolio of risk premia on the Swiss market. Tracking error is also a concern for him.

## Data
To build our factors for our portfolio construction, we first determined all companies which have been listed from January 2009 to November 2021 in the Swiss Performance Index. We collected thenfrom Reuters Datastream the following metrics of these constituents of the SPI from January 2000 to November 2021 (monthly data):
* Price
* Price-to-Earnings
* Dividend yield
* Market Cap
* Beta
* Volatility (unused)
* ROE (unused)
* ROA (unused)
* Gross Margin 
* EPS
* Volume traded
* Industry Classification
* Market-to-book 
* Investments
* Operating Profitability
  
We then collect all necessary macro data to time the factors and 
perform an analysis of the porfolio. We collected from the Federal Reserve of 
Economic Data the following metrics (monthly data): 
* Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland
* Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for the United States
* CBOE Volatility Index: VIX
* Consumer Price Index: All Items for Switzerland
* Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
* TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill
* 3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
* 12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
* 1-Month London Interbank Offered Rate (LIBOR), based on Swiss Franc

## Methodology
We first considered eight popular factors to build our portfolios of factor:
* Momentum (12-months average return)
* Value (E/P)
* Size (Market cap)
* Profitability (Gross margin)
* Low-Beta (Beta)
* Low-Volatility (12-months rolling variance)
* Dividend (Dividend yield)
* Quality Earnings (EPS)
We take either a long position above or below a certain quantile depending on the factor, and remove illiquid companies.

We then considered four different methodologies to build our portfolio, in which one does not tilt the factors:

* No Factor Tilting:
  - Equal-Risk Contribution
* Factor Tilting:
  - Ridge Regression, based on Ulloa, Giamouridis & Montagu (2012) - Risk Reduction in Style Rotation
  - Momentum of Factors
  - Parametric Weights, based on Brandt & Santa-Clara (2006) - Dynamic Portfolio Selection by Augmenting the Asset Space


