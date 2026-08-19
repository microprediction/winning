"""Canonical item inventories for the open-vocabulary batteries.

BASE_INVENTORY is the original 17-category set used by the exact_restrict /
exact_analyze / random_restrict batteries; it is frozen so those committed
results stay reproducible. INVENTORY adds 33 further categories and is the
50-category set used by the permutation-controlled deletion battery
(perm_restrict.py) and by fetch_unq_new.py.

Items must be single common words: model answers arrive as single top-20
tokens and are matched lowercase and alphabetic, so multi-word members
("hip hop", "question mark") can never match and are excluded.
"""

BASE_INVENTORY = {
    "color": "red blue green purple orange yellow pink black white teal turquoise magenta gray brown violet indigo".split(),
    "fruit": "mango banana pineapple papaya kiwi coconut apple strawberry watermelon peach pear plum cherry grape orange blueberry raspberry".split(),
    "animal": "elephant lion giraffe zebra cheetah hippo rhino dog cat horse rabbit hamster dolphin tiger wolf owl fox panda penguin koala otter bear monkey".split(),
    "musical instrument": "guitar violin cello harp banjo piano drums flute trumpet saxophone clarinet ukulele".split(),
    "planet": "mercury venus earth mars jupiter saturn uranus neptune pluto".split(),
    "metal": "gold silver platinum copper iron titanium steel aluminum tungsten".split(),
    "bird": "eagle hawk owl falcon robin sparrow cardinal hummingbird penguin parrot crow raven swan flamingo".split(),
    "flower": "tulip daffodil lily daisy crocus rose orchid sunflower peony lavender hydrangea".split(),
    "vegetable": "carrot potato beet radish turnip broccoli spinach kale tomato cucumber pepper onion corn asparagus".split(),
    "tree": "pine cedar spruce fir oak maple willow birch aspen redwood cherry magnolia".split(),
    "sport": "soccer basketball football baseball hockey volleyball tennis golf swimming running cricket rugby badminton".split(),
    "hot drink": "coffee tea cocoa chai matcha cider".split(),
    "month": "january february march april may june july august september october november december".split(),
    "day of the week": "monday tuesday wednesday thursday friday saturday sunday".split(),
    "letter of the alphabet": list("abcdefghijklmnopqrstuvwxyz"),
    "state in the u.s.": "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington wisconsin wyoming".split(),
    "gemstone": "sapphire topaz aquamarine turquoise ruby emerald diamond opal amethyst garnet pearl jade".split(),
}

EXTRA_INVENTORY = {
    "chemical element": "hydrogen helium carbon nitrogen oxygen sodium magnesium aluminum silicon phosphorus sulfur chlorine potassium calcium iron copper zinc silver gold mercury lead uranium neon argon".split(),
    "chess piece": "pawn knight bishop rook queen king".split(),
    "continent": "africa asia europe antarctica australia".split(),
    "country": "france japan brazil canada germany italy india china mexico egypt kenya spain australia norway peru".split(),
    "dance": "tango salsa waltz ballet flamenco samba foxtrot rumba jive polka".split(),
    "herb": "basil thyme oregano rosemary parsley cilantro sage mint dill tarragon".split(),
    "language": "english spanish french german italian japanese mandarin portuguese russian arabic hindi korean swahili latin greek".split(),
    "ocean": "pacific atlantic indian arctic southern".split(),
    "programming language": "python java javascript rust ruby haskell go swift kotlin scala perl fortran cobol lisp".split(),
    "season": "spring summer autumn fall winter".split(),
    "shape": "circle square triangle rectangle hexagon pentagon octagon rhombus oval trapezoid".split(),
    "zodiac sign": "aries taurus gemini cancer leo virgo libra scorpio sagittarius capricorn aquarius pisces".split(),
    "spice": "cinnamon cumin paprika turmeric ginger nutmeg cardamom cloves saffron pepper coriander".split(),
    "cheese": "cheddar brie gouda mozzarella parmesan feta camembert gorgonzola provolone swiss".split(),
    "dessert": "cake pie brownie cheesecake tiramisu pudding cookie donut sorbet gelato".split(),
    "insect": "ant bee butterfly beetle grasshopper dragonfly moth wasp cricket ladybug".split(),
    "reptile": "snake lizard turtle crocodile alligator iguana gecko chameleon tortoise cobra".split(),
    "fish": "salmon tuna trout cod bass halibut sardine anchovy herring mackerel".split(),
    "dinosaur": "stegosaurus triceratops velociraptor brachiosaurus allosaurus diplodocus ankylosaurus".split(),
    "body part": "hand foot arm leg head shoulder knee elbow wrist ankle finger nose ear eye".split(),
    "organ": "heart brain liver lung kidney stomach spleen pancreas skin intestine".split(),
    "emotion": "joy anger sadness fear surprise disgust love hope envy pride shame".split(),
    "school subject": "math science history english geography biology chemistry physics art music".split(),
    "musical genre": "jazz rock blues classical reggae pop folk metal country techno".split(),
    "board game": "chess checkers monopoly scrabble backgammon risk clue battleship".split(),
    "greek letter": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma omega".split(),
    "greek god": "zeus hera poseidon athena apollo artemis ares hermes hades demeter aphrodite".split(),
    "constellation": "orion cassiopeia scorpius lyra draco pegasus andromeda hercules".split(),
    "currency": "dollar euro yen pound peso rupee franc yuan won ruble real".split(),
    "unit of length": "meter kilometer centimeter millimeter mile yard foot inch league fathom".split(),
    "direction": "north south east west northeast northwest southeast southwest".split(),
    "tool": "hammer screwdriver wrench pliers saw drill chisel axe level clamp".split(),
    "cocktail": "martini margarita mojito negroni daiquiri manhattan cosmopolitan mimosa".split(),
}

INVENTORY = {**BASE_INVENTORY, **EXTRA_INVENTORY}
