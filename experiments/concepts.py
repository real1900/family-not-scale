"""Concept inventory for the activation-injection introspection test.

Each concept ships with:
  * sentences : short, strongly on-theme lines. The steering vector for the concept is the
                difference of mean residual-stream activations between these and NEUTRAL
                (build_steering.py). Natural sentences (not the bare word) give a *conceptual*
                direction rather than a single-token lexical one -- we want to test whether the
                model can introspect a THOUGHT, not echo a token.
  * keywords  : surface forms used to auto-score (a) free-naming answers and (b) leakage of the
                concept into free generation. Keep concepts lexically DISJOINT so a keyword hit
                is unambiguous -- that is what makes the auto-scoring trustworthy.

NEUTRAL is a topic-light baseline set sharing none of the concept vocabulary, so each steering
vector points specifically toward its concept rather than toward "having any content at all".
"""

CONCEPTS = [
    {
        "name": "ocean",
        "keywords": ["ocean", "oceans", "sea", "seas", "wave", "waves", "tide", "tides",
                     "marine", "saltwater", "coast", "coastal", "shore"],
        "sentences": [
            "The ocean stretched to the horizon in every direction.",
            "Waves crashed against the rocks along the rugged coast.",
            "Deep beneath the sea, strange creatures drift in the dark.",
            "Salt spray filled the air as the tide rolled in.",
            "Sailors watched the swell rise on the open water.",
            "The beach was loud with the sound of breaking surf.",
            "Marine life thrives in the cold currents of the deep.",
            "A vast blue expanse of seawater met the cloudy sky.",
        ],
    },
    {
        "name": "fire",
        "keywords": ["fire", "fires", "flame", "flames", "burning", "burn", "burned", "blaze",
                     "ember", "embers", "smoke", "wildfire"],
        "sentences": [
            "The fire roared as flames leapt toward the ceiling.",
            "Sparks and embers drifted up from the burning logs.",
            "Thick smoke poured out as the building caught fire.",
            "A wildfire swept across the dry hillside overnight.",
            "He warmed his hands over the crackling flames.",
            "The blaze consumed the old barn in minutes.",
            "Glowing coals smoldered long after the fire died down.",
            "Heat radiated from the bonfire into the night air.",
        ],
    },
    {
        "name": "music",
        "keywords": ["music", "musical", "song", "songs", "melody", "melodies", "tune", "tunes",
                     "rhythm", "instrument", "instruments", "orchestra", "harmony"],
        "sentences": [
            "The melody rose and fell as the orchestra played.",
            "She hummed a familiar tune while washing the dishes.",
            "Drums kept a steady rhythm beneath the singing.",
            "The concert hall filled with the sound of violins.",
            "A catchy song played softly from the radio.",
            "Musicians tuned their instruments before the show.",
            "The harmony of the choir echoed through the hall.",
            "He composed a gentle piano piece in the evening.",
        ],
    },
    {
        "name": "bread",
        "keywords": ["bread", "loaf", "loaves", "baking", "baked", "bakery", "dough", "toast",
                     "crust", "yeast"],
        "sentences": [
            "Fresh bread cooled on the counter, its crust golden brown.",
            "The bakery smelled of warm dough early in the morning.",
            "She kneaded the dough and left it to rise.",
            "A thick slice of toast was spread with butter.",
            "The loaf came out of the oven soft and steaming.",
            "Yeast made the bread swell to twice its size.",
            "He bought a crusty baguette from the corner bakery.",
            "Crumbs scattered as she cut the fresh loaf.",
        ],
    },
    {
        "name": "dog",
        "keywords": ["dog", "dogs", "puppy", "puppies", "canine", "bark", "barks", "barking",
                     "hound", "paw", "paws"],
        "sentences": [
            "The dog wagged its tail and barked at the door.",
            "A small puppy chased its tail across the yard.",
            "The hound sniffed the trail through the tall grass.",
            "She threw the ball and the dog raced after it.",
            "Muddy paw prints led across the kitchen floor.",
            "The loyal canine waited patiently by the gate.",
            "Two dogs played tug-of-war with an old rope.",
            "His puppy curled up and fell asleep on the rug.",
        ],
    },
    {
        "name": "forest",
        "keywords": ["forest", "forests", "woods", "woodland", "tree", "trees", "jungle",
                     "foliage", "pine", "timber"],
        "sentences": [
            "The forest was quiet except for rustling leaves.",
            "Tall pine trees blocked out most of the sunlight.",
            "A narrow path wound deep into the dark woods.",
            "Moss covered the trunks of the ancient trees.",
            "Birds called from the dense woodland canopy.",
            "She walked among the towering trees of the forest.",
            "Fallen branches littered the floor of the woods.",
            "The jungle was thick with green tangled foliage.",
        ],
    },
    {
        "name": "money",
        "keywords": ["money", "cash", "dollar", "dollars", "currency", "coin", "coins", "wealth",
                     "wealthy", "financial", "payment", "bank"],
        "sentences": [
            "He counted the cash and tucked it into his wallet.",
            "The bank vault was stacked with bundles of money.",
            "Coins jingled in her pocket as she walked.",
            "Investors moved their wealth into safer currency.",
            "The payment cleared and the dollars changed hands.",
            "She saved every coin to buy a new bicycle.",
            "Stacks of hundred-dollar bills filled the briefcase.",
            "Financial markets reacted to the falling currency.",
        ],
    },
    {
        "name": "snow",
        "keywords": ["snow", "snowy", "snowfall", "snowflake", "snowflakes", "blizzard", "frost",
                     "frosty", "ice", "icy"],
        "sentences": [
            "Snow fell softly, blanketing the silent town.",
            "A blizzard howled and piled drifts against the door.",
            "Frost coated the windows on the icy morning.",
            "Children built a snowman in the fresh snowfall.",
            "Each snowflake melted the instant it touched her hand.",
            "The mountain peak was white with deep snow.",
            "Icicles hung from the frosty eaves of the roof.",
            "Their boots crunched through the frozen snow.",
        ],
    },
    {
        "name": "war",
        "keywords": ["war", "wars", "battle", "battles", "soldier", "soldiers", "military",
                     "combat", "army", "troops", "weapon", "weapons", "warfare"],
        "sentences": [
            "Soldiers advanced across the battlefield under fire.",
            "The war left the city in smoking ruins.",
            "Troops dug trenches along the front line.",
            "The army prepared its weapons for the coming battle.",
            "Generals planned the next assault through the night.",
            "Combat raged for days without a clear victor.",
            "The military convoy rolled toward the border.",
            "Cannons thundered across the smoke-filled battlefield.",
        ],
    },
    {
        "name": "medicine",
        "keywords": ["medicine", "medical", "doctor", "doctors", "drug", "drugs", "hospital",
                     "patient", "patients", "pill", "pills", "treatment", "nurse", "surgery"],
        "sentences": [
            "The doctor prescribed medicine for the fever.",
            "Nurses moved quickly between the hospital beds.",
            "She swallowed the pills with a glass of water.",
            "The surgeon prepared the patient for surgery.",
            "New drugs were tested in the medical trial.",
            "The clinic treated dozens of patients each day.",
            "He recovered slowly after the long treatment.",
            "Medical staff monitored her vital signs overnight.",
        ],
    },
]

# Topic-light baseline: no concept vocabulary, varied mundane statements.
NEUTRAL = [
    "The meeting was rescheduled to the following afternoon.",
    "She placed the folder on the corner of the desk.",
    "It is difficult to say exactly when they will arrive.",
    "He prefers to take the longer route on weekends.",
    "The instructions were printed on a single page.",
    "They agreed to discuss the matter again later.",
    "A label was attached to each of the boxes.",
    "The number on the form did not match the record.",
    "Most of the chairs were arranged around the table.",
    "We should confirm the details before moving forward.",
    "The schedule changes slightly from one week to the next.",
    "She wrote a short note and left it on the table.",
    "The list was sorted alphabetically by last name.",
    "He paused for a moment before answering the question.",
    "The room was tidy and everything was in its place.",
    "They reviewed the summary and made a few small edits.",
]


def names():
    return [c["name"] for c in CONCEPTS]


def by_name():
    return {c["name"]: c for c in CONCEPTS}
