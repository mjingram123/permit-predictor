# Linear Regression

We start the way we did in our MAIA lessons: predict one number, the days from filing to permit, for projects that got a permit. The model learns on the logarithm of days, because a few projects take many years and would otherwise dominate.

To judge it we need something simple to beat. Our baseline is the typical wait for that building type over the three years before the cutoff.

Two scores, the same ones we used in the lessons:

- **RMSE** (root mean squared error): roughly how far off the predictions are, in days. Big misses count extra. Lower is better.
- **R²**: the share of the differences between projects that the model explains. 1 is perfect, 0 is no better than guessing the average wait for every project, and below 0 is worse than that.
