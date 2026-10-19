"""A second opinion from a free vision-language model, on a budget that never reaches a bill.

The detector counts people well and misses the far ones: measured on the labelled frames, its
main mistake is a real game read as "a few people" because players at the far touchline are
too small to find. A vision-language model looking at the same picture does not count boxes,
it reads the scene - and a free one can be asked a few times an hour without paying.

**What it may change, and what it may not.** It may upgrade a check the detector found people
in but too few for a game to *playing*, when it says so with confidence. It never turns a
check empty or turns a game off on its own: on the labelled frames the detector called no
empty pitch a game, so there is nothing for it to fix in that direction, and a model that
can be wrong should not be the only thing standing between a booked pitch and a NOTUSED
verdict. Disagreements in that direction are stored for a person to look at.

**Never a bill.** Only providers with a free tier are listed (`providers.py`); each one's use
is counted in the database and refused well below its free limit (`budget.py`); a 429 puts the
provider to rest until its quota resets; and nothing is retried in a loop. Keys come only from
the environment, never from a file in the repository.

**Pictures of people leave the building.** That is a decision, so the feature is off until
``configs/assist.json`` turns it on, heads are blurred before a picture is sent, and the
picture is shrunk to what the model needs. Free tiers may use what they are sent to improve
their models - read the provider's terms before turning it on.

Modules: `config` (the settings file), `providers` (one HTTP client for OpenAI-compatible
endpoints), `budget` (persistent use counts), `reviewer` (when to ask, what to send, how
the answer is combined with the count).
"""
