import streamlit as st
import spacy
from transformers import MarianMTModel, MarianTokenizer
import torch
import pandas as pd

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
    # Create a list of dictionaries for each token
    for token in doc:
        analysis.append({
            "Token": token.text,
            "POS": token.pos_,
            "Dep": token.dep_,
            "Head": token.head.text
        })
    return analysis

def translate_english_to_irish(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def enforce_vso_with_details(text, nlp, is_irish=False):
    doc = nlp(text)
    process_steps = []
    components = {
        'verb': None,
        'subject': None,
        'objects': [],
        'adverbs': [],
        'prepositions': [],
        'temporal_nps': [],
        'auxiliaries': [],
        'negations': []
    }
    
    # Analysis phase
    process_steps.append(("1. Sentence Analysis", f"Analyzing: '{text}'"))
    
    # Find main components
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            components['verb'] = token
            process_steps.append(("\tVerb Found", f"Main verb: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ == "nsubj":
            components['subject'] = token
            process_steps.append(("\tSubject Found", f"Subject: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ in ["obj", "dobj"]:
            components['objects'].append(token)
            process_steps.append(("\tObject Found", f"Object: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ == "advmod":
            components['adverbs'].append(token)
            process_steps.append(("\tAdverb Found", f"Adverb: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ == "prep":
            components['prepositions'].append(token)
            process_steps.append(("\tPreposition Found", f"Preposition: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ == "npadvmod" or (token.dep_ == "advmod" and token.pos_ == "NOUN"):
            components['temporal_nps'].append(token)
            process_steps.append(("\tTemporal NP Found", f"Temporal noun phrase: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ in ["aux", "auxpass"]:
            components['auxiliaries'].append(token)
            process_steps.append(("\tAuxiliary Found", f"Auxiliary: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
        elif token.dep_ == "neg":
            components['negations'].append(token)
            process_steps.append(("\tNegation Found", f"Negation: '{token.text}' (POS: {token.pos_}, Dep: {token.dep_})"))
    
    if not components['verb']:
        process_steps.append(("Warning", "No main verb found - returning original sentence"))
        return text, process_steps
    
    # Build verb phrase
    process_steps.append(("2. Building Verb Phrase", "Constructing the complete verb phrase"))
    
    verb_phrase = []
    # Add auxiliaries first
    for aux in components['auxiliaries']:
        verb_phrase.insert(0, aux.text)
        process_steps.append(("\tAdded Auxiliary", f"'{aux.text}' added to verb phrase"))
    
    # Add negations
    for neg in components.get('negations', []):
        verb_phrase.append(neg.text)
        process_steps.append(("\tAdded Negation", f"'{neg.text}' added to verb phrase"))
    
    # Add main verb
    verb_phrase.append(components['verb'].text)
    process_steps.append(("\tAdded Main Verb", f"'{components['verb'].text}' added to verb phrase"))
    
    verb_phrase = " ".join(verb_phrase)
    process_steps.append(("Verb Phrase Result", f"Final verb phrase: '{verb_phrase}'"))
    
    # Build subject phrase
    subject_phrase = ""
    if components['subject']:
        subject_phrase = " ".join([t.text for t in components['subject'].subtree])
        process_steps.append(("3. Subject Phrase", f"Subject phrase: '{subject_phrase}'"))
    
    # Build object phrases
    object_phrases = []
    for obj in components['objects']:
        obj_phrase = " ".join([t.text for t in obj.subtree])
        object_phrases.append(obj_phrase)
        process_steps.append(("4. Object Phrase", f"Object phrase: '{obj_phrase}'"))
    
    # Build prepositional phrases
    prep_phrases = []
    for prep in components['prepositions']:
        prep_text = prep.text
        for obj in prep.children:
            if obj.dep_ == "pobj":
                prep_text += " " + " ".join([t.text for t in obj.subtree])
        prep_phrases.append(prep_text)
        process_steps.append(("5. Prepositional Phrase", f"Prepositional phrase: '{prep_text}'"))
    
    # Build adverb phrases
    adverb_phrases = []
    for adv in components['adverbs']:
        adv_phrase = " ".join([t.text for t in adv.subtree])
        adverb_phrases.append(adv_phrase)
        process_steps.append(("6. Adverb Phrase", f"Adverb phrase: '{adv_phrase}'"))
    
    # Build temporal phrases
    temporal_phrases = []
    for np in components['temporal_nps']:
        temp_phrase = " ".join([t.text for t in np.subtree])
        temporal_phrases.append(temp_phrase)
        process_steps.append(("7. Temporal Phrase", f"Temporal phrase: '{temp_phrase}'"))
    
    # Construct VSO sentence
    process_steps.append(("8. Constructing VSO Order", "Arranging components in Verb-Subject-Object order"))
    
    if is_irish:
        components_order = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
    else:
        components_order = [verb_phrase, subject_phrase] + object_phrases + adverb_phrases + temporal_phrases + prep_phrases
    
    vso_sentence = " ".join([c for c in components_order if c]).strip()
    process_steps.append(("\tComponents Order", f"Order: {' > '.join([c for c in components_order if c])}"))
    
    # Final adjustments
    if not vso_sentence:
        return text, process_steps
        
    vso_sentence = vso_sentence[0].upper() + vso_sentence[1:]
    if text.endswith("?"):
        vso_sentence += "?"
    else:
        vso_sentence += ("." if not any(p in text[-2:] for p in ["!", "?"]) else "")
    
    process_steps.append(("9. Final Adjustments", f"Capitalization and punctuation added: '{vso_sentence}'"))
    
    return vso_sentence, process_steps

def english_to_irish_vso_pipeline(english_text, nlp, tokenizer, model):
    results = {}
    
    # Step 1: Analyze English structure
    doc = nlp(english_text)
    analysis = analyze_sentence(doc)
    
    # Step 2: Convert English to VSO with detailed process
    english_vso, conversion_process = enforce_vso_with_details(english_text, nlp)
    
    # Step 3: Machine translate to Irish
    irish_raw = translate_english_to_irish(english_text, tokenizer, model)

    for word in english_vso.split(" "):
        print(word)
        print(translate_english_to_irish(word, tokenizer, model))
    
    # Step 4: Convert Irish to VSO (if needed)
    irish_doc = nlp(irish_raw)
    irish_vso, _ = enforce_vso_with_details(irish_raw, nlp, is_irish=True)
    
    return {
        'english_svo': english_text,
        'english_vso': english_vso,
        'irish_svo': irish_raw,
        'irish_vso': irish_vso,
        'analysis': analysis,
        'conversion_process': conversion_process
    }

def display_conversion_process(process_steps):
    st.subheader("SVO to VSO Conversion Process")
    with st.expander("Show Detailed Conversion Steps"):
        for step in process_steps:
            if step[0].startswith("\t"):
                st.markdown(f"↳ *{step[0].strip()}:* {step[1]}")
            else:
                st.markdown(f"**{step[0]}:** {step[1]}")

def main():
    st.title("Verbify")
    st.markdown("""
    This tool converts English sentences (typically SVO word order) to Irish (typically VSO word order).
    It shows the detailed linguistic analysis, the conversion process to VSO, and the final Irish translation.
    """)
    
    # Initialize models
    with st.spinner("Loading NLP models (this may take a few minutes the first time)..."):
        nlp, tokenizer, model = initialize_models()
    
    # User input
    st.subheader("Input Sentence")
    
    # Initialize session state for the text input if it doesn't exist
    if 'english_text' not in st.session_state:
        st.session_state.english_text = "The cat eats fish."
    
    # Create the text area, using the session state value
    english_text = st.text_area("Enter an English sentence to translate:", 
                              st.session_state.english_text)
    
    # Update session state when the text area changes
    if english_text != st.session_state.english_text:
        st.session_state.english_text = english_text
    
    if st.button("Translate"):
        if not english_text.strip():
            st.warning("Please enter a sentence to translate.")
            return
            
        with st.spinner("Processing..."):
            result = english_to_irish_vso_pipeline(english_text, nlp, tokenizer, model)
        
        # Display results
        st.subheader("Results")
        
        # Display the conversion process
        display_conversion_process(result['conversion_process'])
        
        st.subheader("Linguistic Analysis")
        # Convert the analysis to a pandas DataFrame for nice table display
        df = pd.DataFrame(result['analysis'])
        st.table(df.style.set_properties(**{'text-align': 'left'}))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**English (SVO):**")
            st.info(result['english_svo'])
        with col2:
            st.markdown("**English (VSO):**")
            st.info(result['english_vso'])
        
        st.subheader("Irish Translation")
        st.markdown("**Final Translation:**")
        st.success(result['irish_svo'])

        st.markdown("---")
        st.markdown("### Translation Process Explanation")
        st.markdown("""
        1. **Input Analysis**: The English sentence is parsed to identify parts of speech and dependencies.
        2. **VSO Conversion**: 
           - Identify verb, subject, objects, and other components
           - Reorder components to Verb-Subject-Object structure
           - Handle auxiliaries, negations, and other verb modifiers
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
                # Update the session state when a button is clicked
                st.session_state.english_text = example
                # Rerun the app to update the text area
                st.rerun()

if __name__ == "__main__":
    main()