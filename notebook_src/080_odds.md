# From One Number to Odds

All three models above miss by many months, none beats the simple baseline, and every R² is below zero. Two things went wrong, and neither is the model's fault:

1. **They only learn from finished projects.** On the cutoff date, the recently filed projects that already have a permit are the fast ones. The slow ones are still waiting and get left out, so the models learn that recent projects are quick.
2. **A single number hides the real risk.** Many projects never get a permit at all, and "days to permit" does not exist for them.

So we change the question. For every month a project is still waiting, our model estimates the chance that its permit arrives that month. Chaining the months together gives the chance of having a permit by any date. A project that is still waiting on the cutoff date teaches the model something too: it has not been issued yet. If we have only seen part of its latest month, that month counts as a part month.

We compare three ways of giving odds:

- **Track record:** no machine learning. For each month of waiting, what share of projects of this building type got their permit that month, counting only the last two years of city activity.
- **Odds model:** gradient boosting on the monthly question above, using everything we know about the project.
- **Blend:** the average of the two.

Two scores, both on projects the model never saw:

- **Ranking score:** if you pick one project that got its permit in time and one that did not, how often did the method give the first one better odds? 0.5 is a coin flip, 1.0 is perfect.
- **Odds error:** the average squared gap between the odds given and what happened. Lower is better.

A month here means 30 days, so "within 12 months" is 360 days and "within 24 months" is 720.
