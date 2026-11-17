# Data Cleaning

The raw table mixes real buildings with things we do not want:

- **Revisions** of an earlier application. Their permit number has a middle part that does not end in 0000.
- **Plan-check-only filings.** Developers can ask the city to review zoning early. These filings are never meant to become a permit, so counting them as "never issued" would be wrong.
- **Small structures** filed as new buildings: carports, canopies, signs, trailers, guard booths and so on.

We also sort each real building into an asset class. We read the class from the applicant's own project description, not from the city's "use" column, for a reason shown in the next section.
