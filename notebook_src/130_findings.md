# What We Learned

The numbers below are from city data through mid-November 2026. Rerunning the notebook later will move them slightly.

**Key findings**

1. **Getting a permit for a new building in Los Angeles is slow and far from certain.** Of projects filed in 2022 and 2023, about 28% had a building permit two years later. For projects filed in 2013 to 2015 it was about 70%.
2. **Predicting a single date does not work.** Linear regression, a neural network and gradient boosting all missed by more than a year on a typical error measure (RMSE of 440 to 525 days), and none beat the plain "typical wait for that building type" (383 days). Their R² scores are below zero.
3. **Odds and ranges do work.** Across four separate test periods, the date range our tool calls its 80% range held between 79% and 82% of the time, and its 50% range held between 51% and 54%. The blend ranks projects better than the track record alone (0.57 to 0.65 against 0.50 to 0.58, where 0.5 is a coin flip) and has lower error in two of four periods (8% and 14% lower), with a tie in the other two.
4. **A model can cheat without anyone noticing.** Giving the model four columns that the city fills in during review raised its ranking score for a 24-month permit from 0.64 to 0.73. None of that gain is real.
5. **The slowdown is not unique to Los Angeles.** Seattle shows the same decline over the same years.

**Limitations**

- **The odds lean optimistic when the city keeps slowing.** For projects filed 2019 to 2021 the blend gave an average 50% chance of a permit within 24 months and 37% got one. The ranges held up better than the headline odds.
- **Most of what decides the wait is not in public data:** how fast the applicant answers corrections, financing, redesigns. The ranking scores of 0.6 to 0.65 are modest for that reason, and we tested several other data sources (City Planning cases, the filer's track record, county assessor records, a wider set of permits) that did not change this.
- **Building type comes from the applicant's description.** A hand check of 300 descriptions found the first version of our rules right about 77% of the time; we fixed the rules it flagged, but some descriptions stay ambiguous.
- **Plan-check-only filings are left out** unless a later permit application on the same site connects to them. Projects whose parcel number changed along the way can be missed.
- **Hotels are rare** (about 50 projects), so hotel estimates lean on the record for all building types.
- **The planning approval and construction times are simple lookups** from completed cases and finished buildings. They leave out cases that are still open, so they run a little short.
