# Product catalogue and recommender

## Data

Products come from the Amazon Berkeley Objects (ABO) listings, a public research dataset
(147,702 listings; Creative Commons, attribution required; the original release asks for
non-commercial use). `catalog/build_catalog.py` keeps furniture-type listings, maps ABO
product types to the 15 room-item types used by the detector and planner, and writes
`catalog.jsonl` (published as the Hugging Face dataset `daami/furnisher-advisor-catalog`).

What ABO does not have, and what that means:

- **No prices.** The budget planner works on rough INR price bands per item type
  (`backend/app/recommend/plan.py`), shown as ranges and labelled as estimates. Connecting a
  retailer price feed is the planned replacement.
- **No TVs or kitchen appliances**, so those are detected but not recommended.
- **Mostly Amazon house brands**, so the catalogue is a prototype, not a market survey.

## Colour of a product

Colour measured from the catalogue photo was unreliable: photos sit on off-white
backgrounds with shadows, and some are lifestyle shots, so measured colours drifted to grey
(listed-blue sofas measured grey). The listing's own colour word is therefore the primary
colour (`catalog/colour_names.py`), with the photo measurement kept as `rgb_photo` and used
only when the word is unknown. In the published catalogue 3,681 of 5,287 products (70%) use the
listing colour; the other 1,606 rely on the noisier photo measurement and rank slightly lower
(`rgb_source` marks which).

## Ranking

`backend/app/recommend/harmony.py` scores an item colour against the wall colour in [0, 1]:
neutrals suit most walls, other colours score by hue relation (analogous within 40 degrees,
complementary within 25 of 180), and items too close to the wall colour (CIEDE2000 < 12) are
penalised because they disappear. The rules are a transparent baseline that the
feedback-learning ranker will re-weight.

## Budget planner

For each item missing from the room (room checklist minus detections), choose none or one of
three price tiers to maximise importance x tier value within the budget. Solved exactly with
a dynamic programme over the budget in 100-rupee steps; tested for budget respect, priority
order and exhaustion cases.
