# What We Learned

The numbers below are from city data through mid-November 2026. Rerunning the notebook later will move them slightly.

**Key findings**

1. **Getting a permit for a new building in Los Angeles is slow and far from certain.** Of projects filed in 2022 and 2023, about 28% had a building permit two years later. For projects filed in 2013 to 2015 it was about 70%.
2. **Predicting a single date does not work.** Linear regression, a neural network and gradient boosting all had an RMSE above 14 months (440 to 525 days) and lost to the baseline, the plain typical wait for that building type (383 days). Their R² scores are below zero. Only the middle estimate of our odds model beat the baseline (325 days), and its R² is still close to zero.
3. **Odds and ranges do work.** Across four separate test periods, for projects that got a permit, the real wait fell inside the tool's 80% range between 77% and 82% of the time, and inside its 50% range (optimistic to pessimistic) between 48% and 52% of the time. The blend ranks projects better than the track record alone (ranking score of 0.57 to 0.65 against 0.50 to 0.58, where 0.5 is a coin flip) and has lower odds error in two of four periods (5% and 11% lower), with a tie in the other two.
4. **A model can cheat without anyone noticing.** Giving the odds model four columns that the city fills in during review (the cheating model) raised its ranking score for a 24-month permit from 0.64 to 0.74. None of that gain is real.
5. **The slowdown is not unique to Los Angeles.** Seattle's permit rate also fell over the same years, from a much higher starting point, and has partly recovered since 2021. Los Angeles has stayed low.

**Limitations**

- **The odds lean optimistic when the city keeps slowing.** For projects filed 2019 to 2021 the blend gave an average 54% chance of a permit within 24 months and 37% got one. The ranges held up better than the headline odds.
- **Most of what decides the wait is not in public data:** how fast the applicant answers corrections, financing, redesigns. The ranking scores of 0.6 to 0.65 are modest for that reason, and we tested several other data sources (City Planning cases, the filer's track record, county assessor records, a wider set of permits) that did not change this.
- **Building type comes from the applicant's description.** A hand check of 300 descriptions found our first set of rules right about 77% of the time; we fixed the rules it flagged, but some descriptions stay ambiguous.
- **Plan-check-only filings are left out** unless a later permit application on the same site connects to them. Projects whose parcel number changed along the way can be missed. About 100 projects whose permit is recorded on the filing day, with no earlier filing we could find, are also left out, which makes the permit rates here slightly low.
- **Hotels are rare** (about 50 projects), so hotel estimates lean on the record for all building types.
- **The planning approval and construction times are simple lookups** from completed cases and finished buildings. They leave out cases that are still open, so they run a little short.
