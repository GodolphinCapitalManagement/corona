---
title: "Loan Payment Deferrals Due to COVID-19"
subtitle: "A Case Study"
author: [Gyan Sinha]
date: "2020-06-05"
keywords: [COVID-19, Deferments, Consumer Loans]
lang: "en"
titlepage: true
titlepage-text-color: "FFFFFF"
titlepage-rule-color: "360049"
titlepage-rule-height: 2
titlepage-background: "background.pdf"
logo: "godolphin"
header-includes:
    - \graphicspath{{/home/gsinha/admin/docs/logos}}
abstract: "This report analyzes payment deferments or forebearance as a result of
    COVID-19 related shutdowns in the US. We focus on a
    portfolio of unsecured consumer loans originated by 2 different
    institutions. Our analysis focuses on a few key questions: 
    (1) what is the magnitude of COVID related deferments so far? 
    (2) can we estimate deferment probabilities and quantify uncertainty bounds around it?
    (3) does geography impact deferment rates?
    (4) are there systematic relationships across loans that explain deferment requests?
    (5) can regional labor market trends explain the probability of loan deferment? 
    (6) does the sensitivity to labor market shocks vary by region?
    (7) can we leverage this data to generate longer-term, steady-state  deferment rates based
    on assumptions about the future path of labor markets?"
---






## Executive summary

The results presented here are intended to provide a basis for discussion 
about these questions within a general framework that can
be applied not only to unsecured consumer loans but also more broadly,
to other lending sectors. While the data are still  preliminary and the 
events they capture very recent, our conclusions are based on a rigorous and 
transparent statistical analysis and are presented with confidence 
bounds that respect the uncertainty we are currently living through.

In summarizing our results, we would say that larger loans are at greater risk of 
deferment while higher incomes and more seasoning (the age of the loan) tend 
to reduce the chance of a deferment. Borrowers who rent and are self-employed
are at higher risk of deferment. Loans with 5-year amortization terms are
riskier than those with 3-year terms. All else held equal, the impact of FICO scores
and DTI ratios is ambiguous - in one case, they tend to decrease the risk while 
in the other they tend to raise it, albeit by small amounts. On average, 
a 1 standard-deviation increase in a state's weekly claims (as a percentage of 
the labor force) implies a roughly 12% increase in deferment probabilities, 
although this parameter exhibits substantial variation across states.
Given the current level of deferments and 
their recent weekly pace (about 30 bps per week, down from a peak of 3%
to 4% per week in March), we expect deferments to reach []

These forecasts assume cumulative claims of 
approximately 45 million by the end of the second quarter, from their peak on
April 4th, 2020. While the ultimate impact on valuation will depend on cure rates
from deferment, the estimates presented here establish quasi-lower bounds on 
loan values. **In rough numbers, the current deferment run rate of 
30 bps per week, with no cures at deferment expiration, would imply 
additional annualized default rates (in CDR terms) of 
13.43%.**










## Introduction

Our reasons for undertaking this research project were driven by
practical considerations --- like many other investors in consumer and
mortgage lending, we happen to be long these loans. As such, it is
critical for us to evaluate future losses and prospective returns on
these loans and make assessments about their ``fundamental'' value.
We do this with the explicit recognition of the unprecedented nature
of the COVID shock and the fact that in many ways, we are sailing
through uncharted waters.

A natural question that may arise here is the applicability of the analysis
presented given its narrow focus. While there is a natural
tendency to always seek out more and greater amounts of data, in
practice, investors in most cases, hold narrow subsets of the overall population of
loans. While larger datasets may give us more precise estimates (up to
a point), the fact is that we want to make statements about OUR
portfolio, not a fictional universe which is not owned by anyone in
particular. The challenge then is to employ statistical methods that
allow us to extract information from "small" not "big" data and
turn these into useful insights for decision-making. This is where the
bayesian methods we deploy in this report come in useful since they
explicitly deal with inferential uncertainty in an intrinsic way and
can be used to provide insights in other contexts as well.

There are 3 parts to our project. First, we tackle the analysis by
describing the data set in some detail and present
stratifications of the data by different loan attributes. We also
present the deferment rates within each strata in order to get
intuition around the impact of loan attributes. We then
provide statistics around the labor markets in various states. We look
at the impact of initial claims, starting March 14th (which we peg as
the start of the COVID crisis for our purposes) and through the week
ending May 23, 2020, as a percent
of the total employment in each state at the beginning of March. 
An open question that the modeling seeks to answer is the impact of the
claims variables on deferment rates and whether these can be leveraged into a 
prediction framework going forward. A discussion of the statistical model 
that relates the observed outcome (did the loan defer: Yes/No?) to the 
various loan attributes is provided next. The framework employed is based on 
Survival Analysis, using a hierarchical bayes approach as
in \cite{8328358dab6746d884ee538c687aa0dd} and \cite{doi:10.1198/004017005000000661}. 
In closing this part, we
present and discuss the results across the two institutions,
highlighting any differences in the impact of attributes that emerge.

In the second part of our work, we develop a methodology for
forecasting the path of initial claims at the national and state
levels over the next few months. This analysis is unique in its own
way and leverages a brief descriptive note put out by Federal Reserve
Bank of NY researchers in a blog article. We use the claims forecast
as inputs into the predictions for deferment rates at the end of
second quarter of 2020, which is our forecast horizon.

The third part of the project applies the deferment forecasts
developed in the first and second parts to predict deferment
rates at the end of the second quarter. The methodology is 
simple --- we take as given the set of loans that are already
in deferment and project the share of loans that are likely
to be in deferment roughly 8 weeks out. The combined total
gives us the answer we are looking for.

Before we dive into the details, there are 3 key technical aspects in
this report that are worth highlighting.  First, the use of survival or
hazard models to estimate the marginal deferment probability, as a
function of weeks elapsed since the crisis is key to sensible
projections of deferment \footnote{This is a benefit over and above
the intrinsic gain from using this framework in the context of 
"censored" data where most of the observations have not yet 
experienced a deferment event}. As we show, these marginal
hazards have a very strong "duration" component which impacts
longer-term forecasts of the cumulative amount of deferments we expect
over the next few months. 

Second, we extend the survival model
framework by incorporating parameter hierarchies (within a bayesian
framework) that explicitly account for random variation in the impact
of variables, across state clusters. This allows for the 
possibility of ``unobserved heterogeneity'' in the data by
explicitly modeling a state-specific random variable that interacts
with and modifies the hazards for loan clusters within a state. This 
is an important enhancement since (i) there may be
differences in the composition of the workforce across states that
affects the way in which a given volume of claims affects deferment
rates, and (ii) the borrower base itself may differ across states in both
observable and unobservable ways. We control for the observed
attributes explicitly but the hierarchical framework allows us to
model unobserved factors as well. 

Third, we develop a statistical framework 
to model ``decay'' rates for weekly claims and the role that labor markets 
play in determining deferment rates, building upon ideas first discussed 
by researchers at the NY Fed. The projections from this framework serve as 
inputs to our longer-term deferment forecasts and allows us to model the 
impact of different economic scenarios in the future, an important tool to have
in the arsenal given the considerable uncertainties that still remain
regarding the future path of the economy.

## Data
In Table~\ref{tbl:portfolio_summary}, we provided an overview of our 
data sample. In all, we have 3349 loans
in our data, in roughly a 50/50 split (by count) across the 2 institutions.


\label{tbl:portfolio_summary}


|                |    n |   original_balance |   current_balance |       wac |      age |    fico |    term |     defer |       pct |
|:---------------|-----:|-------------------:|------------------:|----------:|---------:|--------:|--------:|----------:|----------:|
| ('I', 'G0')    |  215 |        4.04038e+06 |       1.20461e+06 | 0.0786344 | 22.1732  | 732.574 | 45.6385 | 0.0955156 | 0.0427689 |
| ('I', 'G1')    |  498 |        8.82588e+06 |       3.07975e+06 | 0.1106    | 22.792   | 705.999 | 48.1773 | 0.11051   | 0.109344  |
| ('I', 'G2')    |  533 |        9.73285e+06 |       3.21901e+06 | 0.145183  | 27.6323  | 689.317 | 48.8623 | 0.182699  | 0.114289  |
| ('I', 'G3')    |  260 |        4.69638e+06 |       1.7063e+06  | 0.206051  | 24.6598  | 683.099 | 49.8525 | 0.198377  | 0.0605809 |
| ('I', 'G4')    |   33 |   769725           |  372750           | 0.250535  | 25.6807  | 690.638 | 52.7404 | 0.176581  | 0.0132342 |
| ('II', 'G0')   |  151 |        3.1062e+06  |       1.94539e+06 | 0.0818596 | 11.5285  | 766.627 | 43.3109 | 0.0265655 | 0.0690699 |
| ('II', 'G1')   |  389 |        6.26698e+06 |       3.87199e+06 | 0.103285  | 12.3649  | 726.713 | 41.7204 | 0.0587865 | 0.137472  |
| ('II', 'G2')   |  442 |        7.83342e+06 |       4.83316e+06 | 0.134385  | 14.0987  | 707.304 | 41.4661 | 0.100014  | 0.171598  |
| ('II', 'G3')   |  347 |        6.66718e+06 |       3.82871e+06 | 0.187676  | 16.8439  | 696.506 | 44.56   | 0.191646  | 0.135936  |
| ('II', 'G4')   |  250 |        4.1736e+06  |       2.57631e+06 | 0.251764  | 16.2308  | 683.353 | 46.3566 | 0.144047  | 0.0914702 |
| ('II', 'G5')   |  154 |        1.37806e+06 |       1.13906e+06 | 0.304071  |  8.88784 | 676.266 | 49.0375 | 0.119535  | 0.0404416 |
| ('II', 'G6')   |   77 |   536150           |  388558           | 0.3182    | 10.7678  | 665.143 | 36      | 0.162868  | 0.0137955 |
| ('I', 'ALL')   | 1539 |        2.80652e+07 |       9.58241e+06 | 0.140639  | 24.7851  | 699.061 | 48.5641 | 0.151092  | 0.340217  |
| ('II', 'ALL')  | 1810 |        2.99616e+07 |       1.85832e+07 | 0.163903  | 13.9405  | 709.229 | 43.3774 | 0.111229  | 0.659783  |
| ('ALL', 'ALL') | 3349 |        5.80268e+07 |       2.81656e+07 | 0.155988  | 17.63    | 705.77  | 45.142  | 0.124791  | 1         |

