# import spacy
# from transformers import MarianMTModel, MarianTokenizer
# import torch

# def initialize_models():
#     # Load spaCy English model
#     try:
#         nlp = spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading spaCy 'en_core_web_sm' model...")
#         spacy.cli.download("en_core_web_sm")
#         nlp = spacy.load("en_core_web_sm")

#     # Load MarianMT English-to-Irish model
#     model_name = "Helsinki-NLP/opus-mt-en-ga"
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     return nlp, tokenizer, model

# def analyze_sentence(doc):
#     print("\n--- Linguistic Analysis ---")
#     print("{:<15} {:<10} {:<10} {:<15}".format("Token", "POS", "Dep", "Head"))
#     print("-" * 45)
#     for token in doc:
#         print("{:<15} {:<10} {:<10} {:<15}".format(
#             token.text, token.pos_, token.dep_, token.head.text
#         ))

# def translate_english_to_irish(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         outputs = model.generate(**inputs)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def enforce_vso(text, nlp, is_irish=False):
#     doc = nlp(text)
    
#     # For Irish, check if the sentence is already in VSO format
#     if is_irish:
#         # Check if first word is a verb and second word is a pronoun (common VSO pattern in Irish)
#         if len(doc) > 1 and doc[0].pos_ == "VERB" and doc[1].pos_ == "PRON":
#             return text  # Already in VSO, return as-is
    
#     verb = None
#     subject = None
#     objects = []
#     adverbs = []
#     prepositions = []
#     temporal_nps = []
    
#     # Find main components
#     for token in doc:
#         if token.dep_ == "ROOT" and token.pos_ == "VERB":
#             verb = token
#         elif token.dep_ == "nsubj":
#             subject = token
#         elif token.dep_ in ["obj", "dobj"]:
#             objects.append(token)
#         elif token.dep_ == "advmod":
#             adverbs.append(token)
#         elif token.dep_ == "prep":
#             prepositions.append(token)
#         elif token.dep_ == "npadvmod" or (token.dep_ == "advmod" and token.pos_ == "NOUN"):
#             temporal_nps.append(token)
    
#     if verb:
#         # Build verb phrase
#         verb_phrase = []
#         for child in verb.children:
#             if child.dep_ in ["aux", "auxpass", "neg", "prt"] or child.pos_ in ["AUX", "PART"]:
#                 if child.dep_ in ["aux", "auxpass"]:
#                     verb_phrase.insert(0, child.text)
#                 else:
#                     verb_phrase.append(child.text)
        
#         verb_phrase.append(verb.text)
#         verb_phrase = " ".join(verb_phrase)
        
#         # Handle Irish special cases
#         if is_irish and verb_phrase.lower() in ["is", "are", "'s"]:
#             verb_phrase = "Tá"
        
#         # Subject phrase
#         subject_phrase = " ".join([t.text for t in subject.subtree]) if subject else ""
        
#         # Handle objects
#         object_phrases = [" ".join([t.text for t in obj.subtree]) for obj in objects]
        
#         # Handle prepositions and their objects
#         prep_phrases = []
#         for prep in prepositions:
#             prep_text = prep.text
#             for obj in prep.children:
#                 if obj.dep_ == "pobj":
#                     prep_text += " " + " ".join([t.text for t in obj.subtree])
#             prep_phrases.append(prep_text)
        
#         # Handle adverbs and temporal NPs
#         adverb_phrases = [" ".join([t.text for t in adv.subtree]) for adv in adverbs]
#         temporal_phrases = [" ".join([t.text for t in np.subtree]) for np in temporal_nps]
        
#         # Build VSO sentence
#         if is_irish:
#             # Irish is usually already VSO, so we just ensure proper ordering
#             components = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
#         else:
#             # English VSO: Verb + Subject + (Objects/Adverbials)
#             components = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
        
#         vso_sentence = " ".join([c for c in components if c]).strip()
        
#         # Handle capitalization and punctuation
#         if not vso_sentence:
#             return text
            
#         vso_sentence = vso_sentence[0].upper() + vso_sentence[1:]
#         if text.endswith("?"):
#             return vso_sentence + "?"
#         return vso_sentence + ("." if not any(p in text[-2:] for p in ["!", "?"]) else "")
    
#     return text

# def english_to_irish_vso_pipeline(english_text, nlp, tokenizer, model):
#     print(f"\nOriginal English (SVO): '{english_text}'")
    
#     # Step 1: Analyze English structure
#     doc = nlp(english_text)
#     analyze_sentence(doc)
    
#     # Step 2: Convert English to VSO
#     english_vso = enforce_vso(english_text, nlp)
#     print(f"\nEnglish in VSO order: '{english_vso}'")
    
#     # Step 3: Machine translate to Irish
#     irish_raw = translate_english_to_irish(english_text, tokenizer, model)
#     print(f"\nIrish Translation: '{irish_raw}'")
    
#     # # Step 4:Translate from English VSO to Irish (if needed)
#     # irish_vso = translate_english_to_irish(english_vso, tokenizer, model)
#     # print(f"Irish in VSO order: '{irish_vso}'")
    
#     return {
#         'english_svo': english_text,
#         'english_vso': english_vso,
#         'irish_svo': irish_raw,
#         # 'irish_vso': irish_vso
#     }

# if __name__ == "__main__":
#     nlp, tokenizer, model = initialize_models()
    
#     test_sentences = [
#     "The cat eats fish.",
#     "She reads a book.",
#     "I will go to Dublin tomorrow.",
#     "Do you understand Irish?",
#     "The children are playing outside.",
#     # New simple test sentences
#     "The dog chases the ball.",
#     "She sings a beautiful song.",
#     "We eat dinner at seven.",
#     "He reads the newspaper every morning.",
#     "They visit their grandparents on Sundays."
# ]
    
#     results = []
#     for sentence in test_sentences:
#         result = english_to_irish_vso_pipeline(sentence, nlp, tokenizer, model)
#         results.append(result)
#         print("\n" + "="*60 + "\n")
    
#     print("\n\n=== FINAL CORRECTED COMPARISON ===")
#     for i, result in enumerate(results):
#         print(f"\nExample {i+1}:")
#         print(f"English SVO: {result['english_svo']}")
#         print(f"English VSO: {result['english_vso']}")
#         print(f"Irish Translation: {result['irish_svo']}")
#         # print(f"Irish VSO: {result['irish_vso']}")

#######################################################################################################
import streamlit as st
import spacy
from transformers import MarianMTModel, MarianTokenizer
import torch

# Initialize models (cached for performance)
@st.cache_resource
def initialize_models():
    # Load spaCy English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy 'en_core_web_sm' model... (This may take a few minutes)")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Load MarianMT English-to-Irish model
    model_name = "Helsinki-NLP/opus-mt-en-ga"
    st.write("Loading translation model...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    return nlp, tokenizer, model

def analyze_sentence(doc):
    analysis = []
    analysis.append("{:<15} {:<10} {:<10} {:<15}".format("Token", "POS", "Dep", "Head"))
    analysis.append("-" * 45)
    for token in doc:
        analysis.append("{:<15} {:<10} {:<10} {:<15}".format(
            token.text, token.pos_, token.dep_, token.head.text
        ))
    return "\n".join(analysis)

def translate_english_to_irish(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def enforce_vso(text, nlp, is_irish=False):
    doc = nlp(text)
    
    # For Irish, check if the sentence is already in VSO format
    if is_irish:
        if len(doc) > 1 and doc[0].pos_ == "VERB" and doc[1].pos_ == "PRON":
            return text
    
    verb = None
    subject = None
    objects = []
    adverbs = []
    prepositions = []
    temporal_nps = []
    
    # Find main components
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token
        elif token.dep_ == "nsubj":
            subject = token
        elif token.dep_ in ["obj", "dobj"]:
            objects.append(token)
        elif token.dep_ == "advmod":
            adverbs.append(token)
        elif token.dep_ == "prep":
            prepositions.append(token)
        elif token.dep_ == "npadvmod" or (token.dep_ == "advmod" and token.pos_ == "NOUN"):
            temporal_nps.append(token)
    
    if verb:
        # Build verb phrase
        verb_phrase = []
        for child in verb.children:
            if child.dep_ in ["aux", "auxpass", "neg", "prt"] or child.pos_ in ["AUX", "PART"]:
                if child.dep_ in ["aux", "auxpass"]:
                    verb_phrase.insert(0, child.text)
                else:
                    verb_phrase.append(child.text)
        
        verb_phrase.append(verb.text)
        verb_phrase = " ".join(verb_phrase)
        
        # Handle Irish special cases
        if is_irish and verb_phrase.lower() in ["is", "are", "'s"]:
            verb_phrase = "Tá"
        
        # Subject phrase
        subject_phrase = " ".join([t.text for t in subject.subtree]) if subject else ""
        
        # Handle objects
        object_phrases = [" ".join([t.text for t in obj.subtree]) for obj in objects]
        
        # Handle prepositions and their objects
        prep_phrases = []
        for prep in prepositions:
            prep_text = prep.text
            for obj in prep.children:
                if obj.dep_ == "pobj":
                    prep_text += " " + " ".join([t.text for t in obj.subtree])
            prep_phrases.append(prep_text)
        
        # Handle adverbs and temporal NPs
        adverb_phrases = [" ".join([t.text for t in adv.subtree]) for adv in adverbs]
        temporal_phrases = [" ".join([t.text for t in np.subtree]) for np in temporal_nps]
        
        # Build VSO sentence
        if is_irish:
            components = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
        else:
            components = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
        
        vso_sentence = " ".join([c for c in components if c]).strip()
        
        # Handle capitalization and punctuation
        if not vso_sentence:
            return text
            
        vso_sentence = vso_sentence[0].upper() + vso_sentence[1:]
        if text.endswith("?"):
            return vso_sentence + "?"
        return vso_sentence + ("." if not any(p in text[-2:] for p in ["!", "?"]) else "")
    
    return text

def english_to_irish_vso_pipeline(english_text, nlp, tokenizer, model):
    results = {}
    
    # Step 1: Analyze English structure
    doc = nlp(english_text)
    analysis = analyze_sentence(doc)
    
    # Step 2: Convert English to VSO
    english_vso = enforce_vso(english_text, nlp)
    
    # Step 3: Machine translate to Irish
    irish_raw = translate_english_to_irish(english_text, tokenizer, model)
    
    # Step 4: Convert Irish to VSO (if needed)
    irish_doc = nlp(irish_raw)
    irish_vso = enforce_vso(irish_raw, nlp, is_irish=True)
    
    return {
        'english_svo': english_text,
        'english_vso': english_vso,
        'irish_svo': irish_raw,
        'irish_vso': irish_vso,
        'analysis': analysis
    }

def main():
    st.title("English to Irish VSO Translation Pipeline")
    st.markdown("""
    This tool converts English sentences (typically SVO word order) to Irish (typically VSO word order).
    It shows the linguistic analysis, the conversion to VSO in English, and the final Irish translation.
    """)
    
    # Initialize models
    with st.spinner("Loading NLP models (this may take a few minutes the first time)..."):
        nlp, tokenizer, model = initialize_models()
    
    # User input
    st.subheader("Input Sentence")
    english_text = st.text_area("Enter an English sentence to translate:", 
                              "The cat eats fish.")
    
    if st.button("Translate"):
        if not english_text.strip():
            st.warning("Please enter a sentence to translate.")
            return
            
        with st.spinner("Processing..."):
            result = english_to_irish_vso_pipeline(english_text, nlp, tokenizer, model)
        
        # Display results
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**English (SVO):**")
            st.info(result['english_svo'])
        with col2:
            st.markdown("**English (VSO):**")
            st.info(result['english_vso'])
        
        st.subheader("Linguistic Analysis")
        st.text(result['analysis'])
        
        st.subheader("Irish Translation")
        st.markdown("**Direct Translation (SVO):**")
        st.success(result['irish_svo'])
        
        st.markdown("---")
        st.markdown("### Translation Process Explanation")
        st.markdown("""
        1. **Input Analysis**: The English sentence is parsed to identify parts of speech and dependencies.
        2. **VSO Conversion**: The sentence is restructured to Verb-Subject-Object order while maintaining meaning.
        3. **Machine Translation**: The original English is translated to Irish using a neural machine translation model.
        4. **VSO Adjustment**: The Irish translation is checked and adjusted for proper VSO word order.
        """)
    
    # Example sentences
    st.markdown("---")
    st.subheader("Example Sentences")
    examples = [
        "The cat eats fish.",
        "She reads a book.",
        "I will go to Dublin tomorrow.",
        "Do you understand Irish?",
        "The children are playing outside.",
        "The dog chases the ball.",
        "She sings a beautiful song."
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example):
                st.session_state.english_text = example

if __name__ == "__main__":
    main()