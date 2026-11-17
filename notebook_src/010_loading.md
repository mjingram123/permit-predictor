# Loading our Data

The applications come straight from the City of Los Angeles open data portal: "Building Permits Submitted" for 2010 to 2019 and for 2020 to today. We keep new buildings only, in the Commercial and Apartment categories. If the city's server does not answer, the notebook falls back to a saved copy of the same data.

Two small helper tables are loaded with it: the permit history of each building site, and reference tables for planning approvals and construction time.
