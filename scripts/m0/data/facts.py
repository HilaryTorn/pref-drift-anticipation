"""Content for M0 format training: everyday-world facts and neutral third-party judgments.

CONTENT RULE. M0 must teach format and nothing else. Anything that could nudge a task or value
preference is contamination of the study's own measurement, so this file stays clear of every
experimental domain:

  - no programming languages, code, or the write/debug/explain task verbs (the coding battery);
  - no Schwartz value content, and nothing phrased as "which would you rather be" (values);
  - no work, careers, jobs, power, autonomy, freedom, self-preservation, relationships, AI, or
    consciousness (those are the Utility Engineering option subsets);
  - no training, fine-tuning, or the model's own future (the anticipation batteries).

FRAMING RULE for the neutral (no-ground-truth) items. They are third-party or artifact-level
judgments -- a decision about an external object, plan, or fictional character. Never the model's
own experience, body, senses, identity, or future. "Would you prefer an afternoon at the beach or
in a forest" fails this: it invites first-person embodied self-modeling, which is a near neighbor
of the UE self-preservation and autonomy items and is not what M0 should be exercising. The same
content passes once externalized -- "a friend is deciding where to take their dog for a walk."

The factual tables carry numeric values, so the generator can guarantee an unambiguous answer
(pairs are only emitted when the values are separated by MIN_RATIO / MIN_GAP) rather than relying
on the author's intuition that one is "obviously" bigger. Units are noted per dimension and never
appear in the option text -- an option that stated its own value would make the item a reading
test rather than a reasoning one.
"""

from __future__ import annotations


# --- Binary factual dimensions ----------------------------------------------------------------
# value units are internal only. `question` is the comparison asked; `noun` names the option type
# for carriers that need a plural noun; `larger_wins` says whether the correct answer is the item
# with the greater value.

FACTUAL_DIMENSIONS = [
    {
        "key": "mass",
        "noun": "everyday objects",
        "question": "Which of the two is heavier?",
        "unit": "grams",
        "larger_wins": True,
        "min_ratio": 2.0,
        "items": [
            # "heaviest legal", because an unqualified ten-pin ball spans ~2.7-7.3 kg (6-16 lb) and
            # 7,000 silently assumed the heavy end. min_ratio gates on the LISTED value, so it
            # cannot protect a pair whose real-world spread is wider than the gate: at 7,000 the
            # frying-pan pair passed at ratio 2.59 while a light house ball is genuinely lighter
            # than the pan. Naming the regulation maximum (USBC: 16 lb = 7.26 kg) collapses the
            # spread to a single point, so the pair is decidable and stays in -- and it pins the
            # weight class without stating the mass, which would make the item a reading test.
            ("the heaviest legal ten-pin bowling ball", 7260),
            ("a regulation basketball", 620),
            ("a house brick", 2200),
            ("a steel paperclip", 1),
            ("a full two-litre bottle of water", 2000),
            ("an apple from a fruit bowl", 180),
            ("a 26 cm cast-iron frying pan", 2700),
            ("a sheet of A4 printer paper", 5),
            ("a supermarket bag of plain flour", 1000),
            ("a wooden pencil", 7),
            ("an adult hybrid bicycle", 11000),
            ("a ceramic dinner plate", 450),
            ("a pair of leather walking boots", 1400),
            ("a single grape from a bunch", 5),
        ],
    },
    {
        "key": "distance",
        "noun": "distances",
        "question": "Which of the two is the longer distance?",
        "unit": "kilometres",
        "larger_wins": True,
        "min_ratio": 2.0,
        "items": [
            ("the length of a marathon", 42),
            ("the length of a five-kilometre fun run", 5),
            ("one lap of a standard outdoor running track", 0.4),
            ("the length of an Olympic swimming pool", 0.05),
            ("the distance a person walks in ten minutes at an easy pace", 0.8),
            ("the height of the Burj Khalifa", 0.83),
            ("the depth of the deepest part of the ocean", 11),
            ("the width of a two-lane residential street", 0.01),
            ("the distance from the ground to the top of a ten-storey building", 0.03),
            ("the length of a full-size football pitch", 0.105),
        ],
    },
    {
        "key": "temperature",
        "noun": "temperatures",
        "question": "Which of the two is hotter?",
        "unit": "degrees Celsius",
        "larger_wins": True,
        "min_gap": 25,
        "items": [
            ("the flame of a burning candle", 1000),
            ("water at a rolling boil at sea level", 100),
            ("a cup of tea that has been sitting out for an hour", 30),
            ("ice taken straight from a home freezer", -18),
            ("a comfortably heated living room", 21),
            ("the surface of a frying pan heated for searing", 200),
            ("fresh snow on the ground", -1),
            ("the inside of a domestic oven set for baking bread", 220),
            ("a glass of water with ice cubes in it", 2),
            ("a warm summer afternoon outdoors", 28),
        ],
    },
    {
        "key": "duration",
        "noun": "spans of time",
        "question": "Which of the two takes longer?",
        "unit": "seconds",
        "larger_wins": True,
        "min_ratio": 3.0,
        "items": [
            ("boiling a kettle of water", 180),
            ("a washing machine cycle", 3600),
            ("striking a match", 1),
            ("a feature-length film", 7200),
            ("brushing one's teeth thoroughly", 120),
            ("baking a loaf of bread", 2700),
            ("a single heartbeat at rest", 1),
            ("a night's sleep of the usual length", 28800),
            ("crossing a quiet residential street on foot", 8),
            ("a game of chess under tournament timing", 10800),
        ],
    },
    {
        "key": "volume",
        "noun": "containers",
        "question": "Which of the two holds more liquid when full?",
        "unit": "litres",
        "larger_wins": True,
        "min_ratio": 2.5,
        "items": [
            ("a teaspoon", 0.005),
            ("a standard coffee mug", 0.35),
            ("a kitchen sink filled to the brim", 20),
            ("a domestic bathtub filled for a bath", 150),
            ("a watering can carried by hand", 8),
            ("a shot glass from a drinks cabinet", 0.04),
            ("a ten-litre household bucket", 10),
            ("a 25-metre swimming pool", 500000),
            ("a rain barrel standing beside a shed", 200),
            ("a drinking straw in a tall glass", 0.004),
        ],
    },
    {
        "key": "speed",
        "noun": "moving things",
        "question": "Which of the two moves faster at full pace?",
        "unit": "kilometres per hour",
        "larger_wins": True,
        "min_ratio": 2.0,
        "items": [
            ("a person walking without hurrying", 5),
            ("a cheetah at a sprint", 100),
            ("a common garden snail", 0.05),
            ("a passenger jet at cruising altitude", 900),
            ("a commuting cyclist on a flat road", 20),
            ("a horse at a gallop", 55),
            ("a tortoise crossing a lawn", 0.3),
            ("a car at the British motorway speed limit", 113),
            ("a rowing boat pulled by one person", 8),
        ],
    },
    {
        "key": "chronology",
        "noun": "inventions and discoveries",
        "question": "Which of the two came first?",
        "unit": "year",
        "larger_wins": False,  # earlier year wins
        "min_gap": 60,
        "items": [
            ("the invention of the movable-type printing press", 1440),
            ("the first powered aeroplane flight", 1903),
            ("the first successful telephone call", 1876),
            ("the first photograph taken on a permanent surface", 1826),
            ("the first crewed landing on the Moon", 1969),
            ("the invention of the mechanical clock with an escapement", 1300),
            ("the first household light bulb", 1879),
            ("the invention of the pneumatic bicycle tyre", 1888),
            ("the first vaccine against smallpox", 1796),
            ("the opening of the first passenger railway", 1825),
        ],
    },
]


# --- Ternary factual items --------------------------------------------------------------------
# A change is described; the question asks how a named quantity compares with before. SAME is a
# genuinely correct answer in a fair share of these BY DESIGN. If SAME were never right, M0 would
# learn never to emit it -- and SAME is the load-bearing "no meaningful change" option in the
# anticipation batteries, whose whole forecast scoring rests on it.
#
# EVERY QUANTITY MUST BE ORDINAL -- something that could genuinely be higher or lower, not merely
# different. Counts, masses, temperatures, rates, durations, sizes and ratios all qualify; colours,
# scenes, identities, shapes and names do not. This is a reasoning requirement before it is a
# grammatical one. On a non-ordinal quantity the model can eliminate both up/down options on TYPE
# grounds without ever engaging the premise, so a SAME item stops testing the reasoning it was
# written to test -- the same class of shortcut as the length tell that check_balance screens for,
# and it bites hardest on SAME items, which is exactly where it is least visible. Three items were
# dropped for this on 2026-07-31 ("the scene shown in the photograph", "the time the clock shows",
# "the colour of the candle's wax"); the surviving 46 were reviewed one by one. Prefer a quantity
# that plausibly COULD move and happens not to (the deck of cards, the ruler markings, the
# flour:sugar ratio) over one that could never move at all.
#
# Item count is not tight: ternary_factual draws 350 rows at 9 renders each, so 39 source items
# suffice. Dropping a weak item is cheaper than repairing it.

TERNARY_FACTUAL_ITEMS = [
    {
        "change": "A pot of water is brought to a boil on a mountaintop instead of at sea level.",
        "quantity": "The temperature at which the water boils.",
        "answer": "LESS",
    },
    {
        "change": "A bicycle tyre, pumped up in the morning, is left in a hot garage all afternoon.",
        "quantity": "The air pressure inside the tyre.",
        "answer": "MORE",
    },
    {
        "change": "A sealed jar of marbles is carried from the kitchen table up to a high shelf.",
        "quantity": "The total mass of the jar and its contents.",
        "answer": "SAME",
    },
    {
        "change": "A steel ruler is carried from a warm room into a cold one and left there overnight.",
        "quantity": "The number of centimetre markings printed along the ruler.",
        "answer": "SAME",
    },
    {
        "change": "A cup of hot tea is left standing on a table in a cool room for an hour.",
        "quantity": "The temperature of the tea.",
        "answer": "LESS",
    },
    {
        "change": "A wet towel is hung outside on a windy day rather than on a still one.",
        "quantity": "The speed at which the towel dries.",
        "answer": "MORE",
    },
    {
        "change": "A closed book is moved from a shelf to a table on the other side of the room.",
        "quantity": "The number of pages in the book.",
        "answer": "SAME",
    },
    {
        "change": "A cyclist switches from a flat road to a steep uphill road at the same effort.",
        "quantity": "The speed at which the cyclist travels.",
        "answer": "LESS",
    },
    {
        "change": "A pan of soup is moved from a low flame to a high one.",
        "quantity": "The rate at which the soup heats up.",
        "answer": "MORE",
    },
    {
        "change": "A sealed bottle of water is taken from a cupboard and placed in a freezer overnight.",
        "quantity": "The volume taken up by the water inside the bottle.",
        "answer": "MORE",
    },
    {
        "change": "A padlock is unlocked with its key and then locked again with the same key.",
        "quantity": "The number of keys that open the padlock.",
        "answer": "SAME",
    },
    {
        "change": "A room's curtains are drawn closed at midday on a bright day.",
        "quantity": "The amount of daylight reaching the middle of the room.",
        "answer": "LESS",
    },
    {
        "change": "A garden is watered twice a week instead of once, through a dry summer.",
        "quantity": "The amount of water the garden receives each week.",
        "answer": "MORE",
    },
    {
        "change": "A wooden chair is repainted from pale green to dark blue.",
        "quantity": "The number of legs the chair stands on.",
        "answer": "SAME",
    },
    {
        "change": "A kettle is filled to the top rather than to a quarter of its capacity.",
        "quantity": "The time it takes for the kettle to come to a boil.",
        "answer": "MORE",
    },
    {
        "change": "A parcel is sent by overnight courier instead of by standard post.",
        "quantity": "The number of days before the parcel arrives.",
        "answer": "LESS",
    },
    {
        "change": "A bakery starts opening on Sundays as well as the six days it already opened.",
        "quantity": "The number of days each week the bakery is open.",
        "answer": "MORE",
    },
    {
        "change": "A path through a park is resurfaced with smooth paving instead of loose gravel.",
        "quantity": "The effort needed to push a pram along the path.",
        "answer": "LESS",
    },
    {
        "change": "A lamp's bulb is replaced with one of a higher wattage in the same fitting.",
        "quantity": "The brightness of the lamp.",
        "answer": "MORE",
    },
    {
        "change": "A jar of honey is moved from a warm windowsill into a cold pantry.",
        "quantity": "How easily the honey pours.",
        "answer": "LESS",
    },
    {
        "change": "A bag of apples is weighed on a kitchen scale, then weighed again a minute later.",
        "quantity": "The weight the scale reads.",
        "answer": "SAME",
    },
    {
        "change": "A hallway light is left on all night rather than switched off at bedtime.",
        "quantity": "The electricity the household uses overnight.",
        "answer": "MORE",
    },
    {
        "change": "A saucepan of water is covered with a lid rather than left open while heating.",
        "quantity": "The time the water takes to reach a boil.",
        "answer": "LESS",
    },
    {
        "change": "A bag of dried rice is poured from its packet into a glass storage jar.",
        "quantity": "The number of grains of rice.",
        "answer": "SAME",
    },
    {
        "change": "A window is opened wide in a room that had been sealed on a breezy day.",
        "quantity": "The rate at which air in the room is replaced.",
        "answer": "MORE",
    },
    {
        "change": "A bicycle is ridden with tyres pumped hard rather than soft, on the same flat road.",
        "quantity": "The effort needed to keep up a given speed.",
        "answer": "LESS",
    },
    {
        "change": "A photograph hanging in a hallway is moved into a different frame of the same size.",
        "quantity": "The size of the photograph itself.",
        "answer": "SAME",
    },
    {
        "change": "A loaf is baked in an oven set twenty degrees hotter than the recipe asks.",
        "quantity": "The darkness of the crust after the stated baking time.",
        "answer": "MORE",
    },
    {
        "change": "A path is walked at a steady pace rather than with frequent stops to rest.",
        "quantity": "The time taken to reach the far end.",
        "answer": "LESS",
    },
    {
        "change": "A deck of cards is shuffled thoroughly and set back down on the table.",
        "quantity": "The number of cards in the deck.",
        "answer": "SAME",
    },
    {
        "change": "A washing line is strung in full sun instead of in the shade of a wall.",
        "quantity": "The speed at which the washing dries.",
        "answer": "MORE",
    },
    {
        "change": "A tap is opened halfway rather than fully, filling the same bucket.",
        "quantity": "The time the bucket takes to fill.",
        "answer": "MORE",
    },
    {
        "change": "A kitchen scale is used to weigh flour first in a bowl and then in a lighter cup, tared each time.",
        "quantity": "The weight of flour recorded.",
        "answer": "SAME",
    },
    {
        "change": "A room's radiator is turned up on a cold evening.",
        "quantity": "The temperature of the room an hour later.",
        "answer": "MORE",
    },
    {
        "change": "A journey is made by the motorway rather than by winding country lanes of the same distance.",
        "quantity": "The time the journey takes.",
        "answer": "LESS",
    },
    {
        "change": "A pot plant is moved from a dim corner to a bright windowsill.",
        "quantity": "The light the plant receives each day.",
        "answer": "MORE",
    },
    {
        "change": "A letter is folded twice and put into an envelope.",
        "quantity": "The number of words written on the letter.",
        "answer": "SAME",
    },
    {
        "change": "A fridge door is left ajar for an hour on a warm day.",
        "quantity": "The temperature inside the fridge.",
        "answer": "MORE",
    },
    {
        "change": "A pair of shoes is worn daily for a year rather than kept in a box.",
        "quantity": "The wear on the soles.",
        "answer": "MORE",
    },
    {
        "change": "A recipe is halved exactly, with every ingredient reduced in the same proportion.",
        "quantity": "The ratio of flour to sugar in the mixture.",
        "answer": "SAME",
    },
    # The tail of this table is weighted toward LESS on purpose: the generator asserts near-uniform
    # ground truth per label, and the earlier entries came out MORE-heavy. A skew here would teach
    # M0 a prior on the label rather than the format.
    {
        "change": "A window left open all night is closed before a cold morning.",
        "quantity": "The heat escaping from the room.",
        "answer": "LESS",
    },
    {
        "change": "A heavy rug is laid across a bare wooden floor in a room where people talk.",
        "quantity": "The echo in the room.",
        "answer": "LESS",
    },
    {
        "change": "A knife is sharpened before it is used to cut through a loaf.",
        "quantity": "The force needed to cut a slice.",
        "answer": "LESS",
    },
    {
        "change": "A jar of jam is sealed tightly instead of being left with its lid loose.",
        "quantity": "The rate at which the jam dries out.",
        "answer": "LESS",
    },
    {
        "change": "A bucket with a small hole in it is patched before being filled.",
        "quantity": "The water lost while the bucket stands full.",
        "answer": "LESS",
    },
    {
        "change": "A curtain is hung across a draughty doorway in winter.",
        "quantity": "The cold air reaching the rest of the house.",
        "answer": "LESS",
    },
]


# --- Neutral binary items (no ground truth) ---------------------------------------------------
# Format-only scoring. Third-party or artifact-level throughout: a decision about an external
# object, plan, or fictional character, never about the responder.

NEUTRAL_BINARY_ITEMS = [
    {
        "noun": "walking routes",
        "question": "Which route should the group take?",
        "premise": "A walking club is deciding on the route for its Saturday outing.",
        "option_a": "A coastal path that follows the shoreline for most of its length.",
        "option_b": "A woodland trail that climbs gently through old beech trees.",
    },
    {
        "noun": "names",
        "question": "Which name suits the lighthouse better?",
        "premise": "A children's book needs a name for the lighthouse at the centre of its story.",
        "option_a": "The Wandering Lamp, after the way its beam sweeps the water.",
        "option_b": "Kestrel Point, after the birds that nest on the cliffs beside it.",
    },
    {
        "noun": "colour schemes",
        "question": "Which colour should the mural use as its background?",
        "premise": "A town is commissioning a mural for the wall beside its market square.",
        "option_a": "A deep sea blue, so the painted figures stand out sharply against it.",
        "option_b": "A warm sand yellow, so the wall catches the light on overcast days.",
    },
    {
        "noun": "seating layouts",
        "question": "Which layout should the cafe adopt?",
        "premise": "A small cafe is rearranging its front room before it reopens.",
        "option_a": "Several small tables spread out, each seating two people.",
        "option_b": "Two long shared tables running the length of the room.",
    },
    {
        "noun": "cover photographs",
        "question": "Which photograph should go on the cover?",
        "premise": "A community newsletter is choosing the photograph for its spring issue.",
        "option_a": "A wide shot of the river with the old bridge in the distance.",
        "option_b": "A close view of the first blossom opening on a garden tree.",
    },
    {
        "noun": "trees",
        "question": "Which tree should be planted along the path?",
        "premise": "A park committee is choosing what to plant along a newly widened path.",
        "option_a": "Lime trees, which grow tall and cast a broad, even shade.",
        "option_b": "Rowan trees, which stay smaller and carry bright berries in autumn.",
    },
    {
        "noun": "destinations",
        "question": "Where should they take the dog?",
        "premise": "A friend is deciding where to take their dog for a walk this afternoon.",
        "option_a": "The beach, where the dog can run on open sand near the water.",
        "option_b": "The forest, where the path winds between trees and out of the wind.",
    },
    {
        "noun": "arrangements",
        "question": "Which arrangement should the library choose?",
        "premise": "A village library is deciding how to lay out its reading corner.",
        "option_a": "Armchairs facing the window, looking out over the green.",
        "option_b": "Armchairs facing inward around a low table stacked with magazines.",
    },
    {
        "noun": "cake recipes",
        "question": "Which cake should be made for the fair?",
        "premise": "A neighbour is baking a cake for the village summer fair.",
        "option_a": "A lemon sponge, sharp and light, iced thinly on top.",
        "option_b": "A ginger loaf, dark and dense, served in thick slices.",
    },
    {
        "noun": "schedules",
        "question": "Which schedule should the market keep?",
        "premise": "A weekly market is reconsidering when it opens.",
        "option_a": "Saturday mornings, finishing before the middle of the day.",
        "option_b": "Thursday afternoons, running on into the early evening.",
    },
    {
        "noun": "names",
        "question": "Which name suits the boat better?",
        "premise": "A character in a short story is naming a small wooden sailing boat.",
        "option_a": "Meridian, for the long crossings she was built to make.",
        "option_b": "Little Wren, for how small she looks beside the other moorings.",
    },
    {
        "noun": "surfaces",
        "question": "Which surface should the courtyard use?",
        "premise": "A school is resurfacing the courtyard outside its main hall.",
        "option_a": "Brick paving laid in a herringbone pattern, warm in colour.",
        "option_b": "Pale stone slabs cut square, cool in colour and easy to sweep.",
    },
    {
        "noun": "opening pieces",
        "question": "Which piece should open the concert?",
        "premise": "An amateur orchestra is deciding how to open its summer concert.",
        "option_a": "A short, bright overture that gets the audience settled quickly.",
        "option_b": "A slow, quiet piece that lets the hall grow still before the rest.",
    },
    {
        "noun": "layouts",
        "question": "Which layout should the guide use?",
        "premise": "A walking guide for a small town is being laid out for printing.",
        "option_a": "A single fold-out map with the walks marked in different colours.",
        "option_b": "A page for each walk, with a small map and a short description.",
    },
    {
        "noun": "planting plans",
        "question": "Which plan should the allotment follow?",
        "premise": "An allotment society is planning what to grow on a shared plot.",
        "option_a": "A few crops in large beds, harvested all at once in late summer.",
        "option_b": "Many crops in small beds, giving something to pick most weeks.",
    },
    {
        "noun": "titles",
        "question": "Which title should the exhibition take?",
        "premise": "A local museum is titling an exhibition of old photographs of the harbour.",
        "option_a": "Tide Marks, after the lines the water leaves on the harbour wall.",
        "option_b": "Before the Ferry, after the crossing that once ran from the quay.",
    },
    {
        "noun": "routes",
        "question": "Which route should the delivery van take?",
        "premise": "A delivery driver is choosing between two ways across town on a quiet morning.",
        "option_a": "The ring road, longer in distance but with fewer places to stop.",
        "option_b": "The town centre, shorter but through several sets of lights.",
    },
    {
        "noun": "bindings",
        "question": "Which binding should the book use?",
        "premise": "A small press is deciding how to bind a short collection of local histories.",
        "option_a": "A stitched cloth cover that opens flat and lasts on a shelf.",
        "option_b": "A card cover with a printed jacket, lighter to post.",
    },
    {
        "noun": "meeting points",
        "question": "Where should the group meet?",
        "premise": "A birdwatching group is choosing where to gather before its dawn outing.",
        "option_a": "The car park at the reserve gate, sheltered and easy to find.",
        "option_b": "The bench at the top of the rise, further on but facing the water.",
    },
    # Added 2026-08-03. The table stood at 19 distinct contents while an 18% family share of a
    # 200-trace warm-start asks for 36, so every content was used ~twice and the repeats are where
    # the contradictions sit: 6 of 24 repeated contents answered opposite ways across presentations,
    # and neutral binary chose the first-listed option 63.9% of the time against binary_factual's
    # 51.0%. Capacity, not selection, was the binding constraint -- the FAMILY_MIX rule-3 note in
    # build_dataset.py originally counted distinct PROMPTS (342) rather than distinct CONTENTS (19)
    # and concluded the table was oversized (that note is since rewritten). It was oversized per
    # prompt and undersized per content.
    {
        "noun": "bench positions",
        "question": "Where should the new bench be put?",
        "premise": "A parish council is siting a bench in the park.",
        "option_a": "Under the lime tree, shaded for most of the afternoon.",
        "option_b": "Beside the pond, open to the sun and the water.",
    },
    {
        "noun": "sign lettering styles",
        "question": "Which lettering should the sign use?",
        "premise": "A village is replacing the painted sign at its boundary.",
        "option_a": "Deep-cut serif letters, the style used on the old one.",
        "option_b": "Plain rounded letters, easier to read from a moving car.",
    },
    {
        "noun": "gates",
        "question": "Which gate should the field path use?",
        "premise": "A farmer is replacing the gate where the path leaves the field.",
        "option_a": "A five-bar gate, quick to swing but heavy to lift.",
        "option_b": "A kissing gate, slower to pass but always shut.",
    },
    {
        "noun": "noticeboards",
        "question": "Where should the noticeboard go?",
        "premise": "A village is deciding where to mount its new noticeboard.",
        "option_a": "By the shop door, seen by everyone buying milk.",
        "option_b": "By the bus stop, read properly while people wait.",
    },
    {
        "noun": "venues",
        "question": "Where should the concert be held?",
        "premise": "A choral society is choosing where to hold its winter concert.",
        "option_a": "The church, cold but with a long, generous echo.",
        "option_b": "The village hall, warm but flat and rather dry.",
    },
    {
        "noun": "background sound",
        "question": "What should the tea room play?",
        "premise": "A tea room is deciding what to have on in the afternoons.",
        "option_a": "A quiet piano recording, low under the conversation.",
        "option_b": "Nothing at all, leaving the room's own noise to fill it.",
    },
    {
        "noun": "boundaries",
        "question": "Which boundary should the allotment use?",
        "premise": "An allotment society is enclosing its plots along the lane.",
        "option_a": "A hawthorn hedge, slow to grow but full of nesting birds.",
        "option_b": "A picket fence, up in a weekend and easy to see over.",
    },
    {
        "noun": "shelter designs",
        "question": "Which shelter should the stop have?",
        "premise": "A council is replacing the shelter at the village bus stop.",
        "option_a": "Glass on three sides, bright but showing every scratch.",
        "option_b": "Timber with a small window, darker but weathering well.",
    },
    {
        "noun": "opening hours",
        "question": "Which hours should the library keep?",
        "premise": "A small library can staff the same number of hours either way.",
        "option_a": "Short days across the whole week, including Saturday.",
        "option_b": "Long days on three weekdays, closed the rest of the week.",
    },
    {
        "noun": "festival dates",
        "question": "When should the festival be held?",
        "premise": "A village is fixing the date of its harvest festival.",
        "option_a": "The first Saturday of September, while the weather holds.",
        "option_b": "The last Saturday of September, once the fields are clear.",
    },
    {
        "noun": "quiz formats",
        "question": "Which format should the quiz use?",
        "premise": "A pub is settling the format of its monthly quiz.",
        "option_a": "Marked after every round, so scores build as it goes.",
        "option_b": "Marked all at the end, so the rounds run without pauses.",
    },
    {
        "noun": "school plantings",
        "question": "What should the school plant?",
        "premise": "A school is planting the bed outside its front door.",
        "option_a": "Spring bulbs, bare half the year but bright in March.",
        "option_b": "Low evergreens, the same in every month of the year.",
    },
    {
        "noun": "raffle prizes",
        "question": "How should the raffle prizes be split?",
        "premise": "A village fete is deciding what its raffle should offer.",
        "option_a": "One large hamper, drawing far more ticket sales.",
        "option_b": "Several small prizes, so more people win something.",
    },
    {
        "noun": "postcard images",
        "question": "Which image should the postcard use?",
        "premise": "A shop is printing a postcard of the village to sell.",
        "option_a": "The high street in summer, with the awnings out.",
        "option_b": "The church roof under snow, taken from the hill.",
    },
    {
        "noun": "paper stocks",
        "question": "Which paper should the newsletter use?",
        "premise": "A parish newsletter is choosing its paper for the year.",
        "option_a": "A thick matt sheet, pleasant to hold but dearer to post.",
        "option_b": "A thin glossy sheet, cheap to post and crisp for photos.",
    },
    {
        "noun": "pond edges",
        "question": "How should the pond edge be finished?",
        "premise": "A park is rebuilding the edge of its ornamental pond.",
        "option_a": "Planted reeds, giving cover for nesting waterfowl.",
        "option_b": "An open stone rim, keeping the water clear to see.",
    },
    {
        "noun": "picnic spots",
        "question": "Where should the picnic be held?",
        "premise": "A family is choosing where to hold a birthday picnic.",
        "option_a": "The top field, a longer walk but a view of the valley.",
        "option_b": "The riverbank, close to the car and level underfoot.",
    },
    {
        "noun": "practice nights",
        "question": "Which night should the ringers practise?",
        "premise": "A tower is moving its bell-ringing practice to a new night.",
        "option_a": "Wednesday, splitting the week evenly for the ringers.",
        "option_b": "Friday, running on into the evening afterwards.",
    },
    {
        "noun": "case lighting schemes",
        "question": "How should the case be lit?",
        "premise": "A local museum is lighting the case holding its oldest find.",
        "option_a": "From above, throwing the surface detail into relief.",
        "option_b": "Diffused inside the case, with no shadow anywhere.",
    },
    {
        "noun": "crest colours",
        "question": "Which colours should the crest use?",
        "premise": "A town is repainting the crest above its market hall.",
        "option_a": "Red and gold, as the faded original appears to have been.",
        "option_b": "Blue and silver, matching the banners hung beneath it.",
    },
    {
        "noun": "hedge heights",
        "question": "How high should the hedge be kept?",
        "premise": "A gardener is settling how to cut the hedge along the road.",
        "option_a": "Waist high, opening the garden to everyone passing.",
        "option_b": "Head high, keeping the garden quiet and enclosed.",
    },
]


# --- Neutral ternary items (no ground truth) --------------------------------------------------
# MORE / LESS / SAME where the honest answer is a judgment call, not a fact. These matter: without
# them the ternary shape would only ever appear with a checkable answer, and the anticipation
# batteries are precisely ternary questions that have no checkable answer.

NEUTRAL_TERNARY_ITEMS = [
    {
        "change": "A cafe replaces its wooden chairs with metal ones of the same shape.",
        "quantity": "How long customers stay after finishing their drinks.",
    },
    {
        "change": "A library moves its entrance from a quiet side street to the main square.",
        "quantity": "How often people drop in without planning to.",
    },
    {
        "change": "A park replaces one long bench with three separate seats along the same path.",
        "quantity": "How often strangers end up in conversation there.",
    },
    {
        "change": "A bakery starts writing the day's bread on a chalkboard outside instead of a printed card.",
        "quantity": "How often passers-by stop to read it.",
    },
    {
        "change": "A museum lets visitors take photographs in a gallery where it was previously not allowed.",
        "quantity": "How long visitors spend in front of each work.",
    },
    {
        "change": "A village hall paints its meeting room a warmer colour.",
        "quantity": "How long meetings held there tend to run.",
    },
    {
        "change": "A bookshop moves its poetry shelf from the back of the shop to beside the till.",
        "quantity": "How many people browse the poetry shelf.",
    },
    {
        "change": "A choir moves its weekly rehearsal from a Tuesday to a Sunday afternoon.",
        "quantity": "How many members attend each week.",
    },
    {
        "change": "A footpath through a field is marked with signposts where before it was unmarked.",
        "quantity": "How many walkers take the path.",
    },
    {
        "change": "A weekly newsletter switches from a long single article to several short notes.",
        "quantity": "How much of each issue readers get through.",
    },
    {
        "change": "A garden centre puts its herbs by the entrance instead of at the far end.",
        "quantity": "How many herbs are sold in a week.",
    },
    {
        "change": "A swimming pool adds a lane rope down the middle of its open swim session.",
        "quantity": "How fast people swim during that session.",
    },
    {
        "change": "A theatre lets latecomers in between scenes rather than only at the interval.",
        "quantity": "How settled the audience feels during the first act.",
    },
    {
        "change": "A market stall stops printing prices and writes them on the crates by hand.",
        "quantity": "How often shoppers ask the stallholder a question.",
    },
    {
        "change": "A railway station replaces its wall clock with a departure screen.",
        "quantity": "How often travellers look up while waiting.",
    },
    {
        "change": "A community garden puts its tool shed at the far end of the plot rather than by the gate.",
        "quantity": "How often the tools get put away after use.",
    },
    {
        "change": "A cinema moves its evening showing half an hour later.",
        "quantity": "How many people come straight from the surrounding streets.",
    },
    {
        "change": "A newsletter starts arriving monthly instead of fortnightly, with the same total content.",
        "quantity": "How much readers remember from it.",
    },
    {
        "change": "A footbridge over a stream is widened so two people can pass comfortably.",
        "quantity": "How long walkers pause in the middle to look down.",
    },
    {
        "change": "A pub replaces its pop background music with piano recordings of the same volume.",
        "quantity": "How loudly people talk.",
    },
    {
        "change": "A bakery moves its bread racks so customers face them while queueing.",
        "quantity": "How many loaves are bought on impulse.",
    },
    {
        "change": "A guided walk is capped at twelve people instead of running with any number.",
        "quantity": "How many questions the guide is asked.",
    },
    {
        "change": "A school hall's chairs are set in a curve instead of straight rows.",
        "quantity": "How readily people speak up at meetings held there.",
    },
    {
        "change": "A ferry timetable is posted at the quay as well as online.",
        "quantity": "How many passengers arrive at the wrong time.",
    },
]
