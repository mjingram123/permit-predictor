## A trap in the data

Never assume a column was filled in on the day of filing. Below we compare how often each column has a value for applications that got a permit against applications that never did.

Some columns are almost always filled in for issued permits and often blank for the rest. The city completes them during review. A model that sees those columns learns "blank means no permit", which looks like great accuracy and is useless for a project that has not been reviewed yet. We keep only the columns that are filled in equally for both groups, and we read building details from the applicant's own description instead.
