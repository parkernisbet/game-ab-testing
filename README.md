# Game AB Testing

This dataset was originally released as a Datacamp [project](https://www.datacamp.com/projects/184), and then later as a Kaggle [dataset](https://www.kaggle.com/yufengsui/mobile-games-ab-testing). It contains player interaction and interval retention information for just over 90000 unique users from a popular mobile puzzle [game](https://tactilegames.com/cookie-cats/). Over the span of a week, new users were fed one of two game progression systems and then monitered for changes in outcome (retention). 

My objective was to construct an A/B test to evaluate the impact of said progression system on player retention. To account for the categorical dependent and independent variables, a two sample chi2 test was enacted to evaluate variable independence. The end result, a p-value less than our pre-defined confidence interval, pointed to a rejection of the null hypothesis. With statistical certainty, this notebook highlights the dependent nature of the two variables under analysis.

One interesting finding in the initial data exploration, the progression system change brought about opposite effects for short and long term retention. Generally speaking, the overall impact on retention was net negative (i.e. adding the new progression system resulted in fewer retained players across all time intervals), however short term retention actually saw an ever so slight improvement. My best guess would be because the new progression system delays a forced waiting period (a very basic description of the implemented change), players who were slower to get to progress to this point would carry over play to the next day after install. 
