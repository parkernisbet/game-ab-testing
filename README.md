# Game AB Testing

This dataset was originally released as a Datacamp [project](https://www.datacamp.com/projects/184), and then later as a Kaggle [dataset](https://www.kaggle.com/yufengsui/mobile-games-ab-testing). It contains player interaction and interval retention information for just over 90000 unique users from a popular mobile puzzle [game](https://tactilegames.com/cookie-cats/). Over the span of a week, new users were fed one of two game progression systems and then monitered for changes in outcome (retention at specific time intervals and game interaction by end of week). 

A series of A/B tests were constructed to evaluate the impact of said progression system on both of the above mentioned KPIs. Based on requriements detailed in the notebook, the tests for statistical significance were as follows:

  - player retention:
    + test: chi2 contingency table
 
    + null: progression system has no impact on retention
 
    + alternative: progression system has an impact on retention

    + outcome: reject null; negative correlation between progression system and retention

  - game interaction:
    + test: wilcoxon rank sum

    + null: game interaction distributions are the same

    + alternative: game interaction distributions are different

    + outcome: accept null; progression system has no impact on game interaction

One interesting finding in the initial data exploration, the progression system change brought about opposite effects for short and long term retention. Generally speaking, the overall impact on retention was net negative (i.e. adding the new progression system resulted in fewer retained players across all time intervals), however short term retention actually saw an ever-so-slight improvement. My best guess would be because the new progression system delays a forced waiting period (a very basic description of the implemented change), players who were slower to reach this point were more likely to carry over play to the next day.
