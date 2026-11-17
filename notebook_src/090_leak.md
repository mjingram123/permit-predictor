# Catching a Model That Cheats

Remember the columns the city fills in during review: building use, floor area, height and construction type. What happens if we hand them to the same odds model?

Its ranking score jumps. It looks like a much better model. It is not. We call it the cheating model: it has learned that a blank floor area means the city never finished reviewing the project. On filing day every project has those columns blank, so the extra "accuracy" disappears the moment the tool is used for real.
