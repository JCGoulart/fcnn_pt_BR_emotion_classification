# Third party imports
import spacy
import emoji # Since the data contains emojis, we can demojize the text to find the meaning of the emojis

# Function to preprocess the data
def preprocess_text(text, nlp):
    """
    Preprocess the text by removing stop words and applying lemmatization.
    
    This functions realize the following steps:
    - Analyze text using Spacy model
    - For each token in the text apply:
        - lematization
        - Convert to lowercase
        - Remove extra spaces
        - Remove stop words

    - Ignore tokes that are stop words
    - Join the tokens into a single string split by space
    - Return the preprocessed text

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.

    """

    # Demojize the text
    text = emoji.demojize(text)

    # Preprocess the text with Spacy model
    doc = nlp(text)

    # Apply lemmatization, lowercase conversion, and stop word removal
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]

    # Join the tokens into a single string
    return ' '.join(tokens)

# Load the spacy model for preprocessing
spacy.cli.download("pt_core_news_md")
nlp = spacy.load("pt_core_news_md")