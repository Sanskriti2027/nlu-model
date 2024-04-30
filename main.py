import spacy
from spacy.matcher import Matcher

# Loading the spacy language model
nlp = spacy.load("en_core_web_sm")

# Defining intents and patterns
intents = {
    "book_flight": [
        ["book a flight to {location}"],
        ["can I get a flight to {location}"]
    ],
    "check_weather": [
        ["what's the weather like in {location}"],
        ["tell me the weather at {location}"]
    ]
}

# Creating a spaCy matcher for entities
matcher = Matcher(nlp.vocab)

# Defining entity patterns
patterns = [
    [{"LOWER": "flight"}],
    [{"LOWER": "weather"}],
    [{"LOWER": "at"}, {"POS": "DET", "OP": "?"}, {"LOWER": "the"}, {"POS": "NOUN"}],
]

# Adding entity patterns to the matcher
for pattern in patterns:
    matcher.add("EntityPattern", [pattern])

# NLU class
class NLU:
    def __init__(self):
        self.matcher = matcher

    def recognize_intent(self, text):
        doc = nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # getting the ID of the matched rule
            span = doc[start:end]  # getting the matched span

            if rule_id in intents:
                intent = rule_id
                entities = {ent.label_: ent.text for ent in span.ents}
                return intent, entities

        return None, {}

# Testing the NLU module
nlu = NLU()

texts = [
    "book a flight to Paris",
    "can I get a flight to Rome",
    "what's the weather like in London",
    "tell me the weather at New York"
]

for text in texts:
    intent, entities = nlu.recognize_intent(text)
    print(f"Text: {text}")
    if intent:
        print(f"Intent: {intent}")
    if entities:
        print(f"Entities: {entities}")
    print()
    
