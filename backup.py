# import spacy

# def identify_svo(sentence):
#     # Load the English language model
#     nlp = spacy.load("en_core_web_sm")
    
#     # Process the sentence
#     doc = nlp(sentence)
    
#     # Initialize variables
#     subject = None
#     verb = None
#     obj = None
    
#     # Iterate through tokens to find SVO
#     for token in doc:
#         # Find subject (usually a noun with nsubj dependency)
#         if "subj" in token.dep_:
#             subject = token.text
#             # Include compound nouns (e.g., "dog food" as subject)
#             for child in token.children:
#                 if child.dep_ == "compound":
#                     subject = f"{child.text} {token.text}"
        
#         # Find main verb (usually ROOT of the sentence)
#         if token.dep_ == "ROOT" and token.pos_ == "VERB":
#             verb = token.text
        
#         # Find object (usually a noun with dobj dependency)
#         if "dobj" in token.dep_:
#             obj = token.text
#             # Include compound nouns for objects too
#             for child in token.children:
#                 if child.dep_ == "compound":
#                     obj = f"{child.text} {token.text}"
    
#     return {
#         "sentence": sentence,
#         "subject": subject,
#         "verb": verb,
#         "object": obj
#     }

# # Example usage
# sentence = input("Enter sentence: ")
# result = identify_svo(sentence)

# # Print in Subject-Object-Verb (SOV) order
# print("\nAnalysis:")
# print(f"Sentence: {result['sentence']}")
# print(f"Subject: {result['subject']}")
# print(f"Object: {result['object']}")
# print(f"Verb: {result['verb']}")

# # Bonus: Print reconstructed sentence in SOV order (if all components found)
# if result['subject'] and result['object'] and result['verb']:
#     print("\nReconstructed (SOV order):")
#     print(f"{result['subject']} {result['object']} {result['verb']}")
# else:
#     print("\nNote: Could not reconstruct full SOV sentence (missing components)")


# import spacy

# def identify_svo(sentence):
#     # Load the English language model
#     nlp = spacy.load("en_core_web_sm")
    
#     # Process the sentence
#     doc = nlp(sentence)
    
#     # Initialize variables
#     subject = None
#     verb = None
#     obj = None
    
#     # Iterate through tokens to find SVO
#     for token in doc:
#         # Find subject (usually a noun with nsubj dependency)
#         if "subj" in token.dep_:
#             subject = token.text
#             # Include compound nouns (e.g., "dog food" as subject)
#             for child in token.children:
#                 if child.dep_ == "compound":
#                     subject = f"{child.text} {token.text}"
        
#         # Find main verb (usually ROOT of the sentence)
#         if token.dep_ == "ROOT" and token.pos_ == "VERB":
#             verb = token.text
        
#         # Find object (usually a noun with dobj dependency)
#         if "dobj" in token.dep_:
#             obj = token.text
#             # Include compound nouns for objects too
#             for child in token.children:
#                 if child.dep_ == "compound":
#                     obj = f"{child.text} {token.text}"
    
#     return {
#         "sentence": sentence,
#         "subject": subject,
#         "verb": verb,
#         "object": obj
#     }

# # Example usage
# # ]sentence = "The quick brown fox jumps over the lazy dog"
# sentence = input("Enter sentence: ")
# result = identify_svo(sentence)
# print(f"Sentence: {result['sentence']}")
# print(f"Subject: {result['subject']}")
# print(f"Verb: {result['verb']}")
# print(f"Object: {result['object']}")

# import spacy
# from spacy.tokens import Token

# # Load English model
# nlp = spacy.load("en_core_web_sm")

# def reorder_sentence(text):
#     doc = nlp(text)
#     reordered = []
#     verb_phrases = {}

#     for token in doc:
#         # Capture the verb and group associated elements
#         if token.pos_ == "VERB":
#             verb = token
#             group = {
#                 "subject": [],
#                 "dobj": [],
#                 "prep_phrases": [],
#                 "verb": token.text,
#             }

#             # Find subject
#             for child in token.children:
#                 if child.dep_ == "nsubj":
#                     group["subject"].append(child.text)
#                 elif child.dep_ == "dobj":
#                     group["dobj"].append(child.text)
#                 elif child.dep_ == "prep":
#                     prep_phrase = [child.text]
#                     for sub in child.children:
#                         prep_phrase.append(sub.text)
#                     group["prep_phrases"].append(" ".join(prep_phrase))

#             verb_phrases[token.i] = group

#     # Build reordered sentence
#     for vp_index in sorted(verb_phrases.keys()):
#         group = verb_phrases[vp_index]
#         parts = []
#         parts.extend(group["subject"])
#         parts.extend(group["dobj"])
#         parts.extend(group["prep_phrases"])
#         parts.append(group["verb"])
#         reordered.append(" ".join(parts))

#     return " ".join(reordered)

# # Example sentences
# examples = [
#     "She eats sushi.",
#     "He put the book on the table.",
#     "They read books in the library.",
#     "The cat chased the mouse under the sofa."
# ]

# for sentence in examples:
#     print(f"Original: {sentence}")
#     print(f"Reordered: {reorder_sentence(sentence)}")
#     print()

# LIORA VERSIO
# import spacy

# # Load the English language model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading spaCy 'en_core_web_sm' model...")
#     spacy.cli.download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# def analyze_sentence(sentence_text):
#     """
#     Parses a sentence using spaCy and prints relevant linguistic information
#     to help in rule annotation.
#     """
#     doc = nlp(sentence_text)

#     print(f"\n--- Analysis for: '{sentence_text}' ---")
#     print("\nTokens and their properties:")
#     print("{:<15} {:<10} {:<10} {:<10} {:<15} {:<10} {:<10}".format(
#         "Token", "POS", "TAG", "DEP", "HEAD", "HEAD_POS", "Children"
#     ))
#     print("-" * 80)
#     for token in doc:
#         children_texts = [child.text for child in token.children]
#         print("{:<15} {:<10} {:<10} {:<10} {:<15} {:<10} {:<10}".format(
#             token.text, token.pos_, token.tag_, token.dep_, token.head.text, token.head.pos_, str(children_texts)
#         ))

#     print("\nDependency Tree (visual, simplified):")
#     for token in doc:
#         if token.dep_ == "ROOT":
#             print(f"[ROOT] '{token.text}' ({token.pos_})")
#         else:
#             print(f"  └─'{token.text}' ({token.pos_}) --({token.dep_})--> '{token.head.text}' ({token.head.pos_})")

#     print("\nNoun Chunks (NPs):")
#     for chunk in doc.noun_chunks:
#         print(f"  - '{chunk.text}' (Root: '{chunk.root.text}', Root Dep: '{chunk.root.dep_}')")

#     print("-" * 80 + "\n")
#     return doc

# # --- Example Usage to Analyze Sentences for Rule Annotation ---

# # Example 1: Simple SVO
# doc1 = analyze_sentence("She eats sushi.")

# # Example 2: SVO with adjective
# doc2 = analyze_sentence("The quick brown fox jumps over the lazy dog.")

# # Example 3: SVO with prepositional phrase
# doc3 = analyze_sentence("I read a book in the library.")

# # Example 4: Sentence with an adverb
# doc4 = analyze_sentence("He quickly ran home.")

# # --- Conceptual Structure for your ANNOTATED RULES ---
# # This is NOT executable code that performs reordering.
# # This is how you would *document* your rules in a structured,
# # machine-readable (by Member 2) format.

# # Option 1: List of Dictionaries (highly structured)
# # This format directly guides Member 2's implementation.
# reordering_rules_data = [
#     {
#         "rule_id": "R1_SVO_to_SOV_DirectObject",
#         "name": "Move Direct Object before Verb",
#         "purpose": "Transforms basic SVO (Subject-Verb-Object) to SOV.",
#         "original_pattern": {
#             "root_verb": {"pos": "VERB", "dep": "ROOT"},
#             "subject": {"pos": ["NOUN", "PRON"], "dep": "nsubj", "head_rel": "root_verb"},
#             "direct_object": {"pos": ["NOUN", "PRON"], "dep": "dobj", "head_rel": "root_verb"}
#         },
#         "reordered_elements_order": ["subject", "direct_object", "root_verb"],
#         "example_input": "She eats sushi.",
#         "example_parse_highlights": {
#             "She": {"pos": "PRON", "dep": "nsubj"},
#             "eats": {"pos": "VERB", "dep": "ROOT"},
#             "sushi": {"pos": "NOUN", "dep": "dobj"}
#         },
#         "example_reordered_output": "She sushi eats.",
#         "notes": "Handles simple noun/pronoun subjects and direct objects."
#     },
#     {
#         "rule_id": "R2_Adverb_Pre_Verb",
#         "name": "Move Adverbial Modifier before Verb",
#         "purpose": "Places adverbs (advmod) before the verb they modify.",
#         "original_pattern": {
#             "root_verb": {"pos": "VERB", "dep": "ROOT"},
#             "adverb": {"pos": "ADV", "dep": "advmod", "head_rel": "root_verb"}
#         },
#         "reordered_elements_order": ["subject_of_root", "adverb", "root_verb", "objects_of_root"], # Need to generalize for other elements
#         "example_input": "He quickly ran home.",
#         "example_parse_highlights": {
#             "He": {"pos": "PRON", "dep": "nsubj"},
#             "quickly": {"pos": "ADV", "dep": "advmod"},
#             "ran": {"pos": "VERB", "dep": "ROOT"}
#         },
#         "example_reordered_output": "He quickly ran home.", # 'home' is a noun, advmod here
#         "notes": "Considers adverbs that modify the main verb."
#     },
#     # Add 3-8 more rules here following the same structure
#     # Consider rules for:
#     # - Adjective placement (before/after noun)
#     # - Prepositional phrase reordering (e.g., location, time)
#     # - Indirect objects (e.g., "give money to him" -> "give him money")
#     # - Handling auxiliary verbs (e.g., "is eating", "will go")
# ]

# # Option 2: Detailed comments/docstrings for each rule (less structured for automation, but clear for human)
# # This would be useful if you're writing a separate document or a Python file
# # that primarily serves as documentation.

# """
# # Rule 1: Simple SVO to SOV (Direct Object Movement)
# # Purpose: Moves a simple direct object (dobj) from after the root verb to before it.
# # Original English Pattern (using spaCy relations):
# #   - ROOT verb (VERB)
# #   - Subject (NOUN/PRON) with 'nsubj' dependency to ROOT
# #   - Direct Object (NOUN/PRON) with 'dobj' dependency to ROOT
# # Reordered Pattern: [Subject] [Direct Object] [Verb]
# # Example:
# #   - Input: "She eats sushi."
# #   - spaCy Analysis (relevant parts):
# #     - Token: She, POS: PRON, DEP: nsubj, HEAD: eats
# #     - Token: eats, POS: VERB, DEP: ROOT
# #     - Token: sushi, POS: NOUN, DEP: dobj, HEAD: eats
# #   - Reordered Output: "She sushi eats."
# """

# """
# # Rule 2: Adverbial Modifier Pre-Verb
# # Purpose: Places adverbs (ADV) that directly modify the main verb (advmod dependency) before the verb.
# # Original English Pattern:
# #   - ROOT verb (VERB)
# #   - Adverb (ADV) with 'advmod' dependency to ROOT
# # Reordered Pattern: [Subject] [Adverb] [Verb] [Other objects/complements]
# # Example:
# #   - Input: "He quickly ran home."
# #   - spaCy Analysis (relevant parts):
# #     - Token: He, POS: PRON, DEP: nsubj, HEAD: ran
# #     - Token: quickly, POS: ADV, DEP: advmod, HEAD: ran
# #     - Token: ran, POS: VERB, DEP: ROOT
# #     - Token: home, POS: NOUN, DEP: advmod, HEAD: ran (Note: 'home' can be advmod here)
# #   - Reordered Output: "He quickly ran home." (or "He home quickly ran" depending on complexity)
# """

# # You would continue this pattern for 5-10 rules.

# # Example of how Member 1 might "test" the rule conceptually
# # This is NOT the preprocessor, but just Member 1 verifying the rule logic
# # If you run this part, you'll see how you can extract elements based on your rules.
# # This helps refine the rule definition for Member 2.

# # --- Conceptual Rule Application (for Member 1's validation) ---
# def conceptual_apply_rule1(doc):
#     """
#     Conceptual application of R1 for validation.
#     Extracts elements based on R1 pattern for 'She eats sushi.'
#     """
#     subject = None
#     verb = None
#     dobj = None

#     for token in doc:
#         if token.dep_ == "nsubj" and token.head.dep_ == "ROOT":
#             subject = token
#         elif token.dep_ == "ROOT" and token.pos_ == "VERB":
#             verb = token
#         elif token.dep_ == "dobj" and token.head == verb: # Ensure dobj is directly linked to the verb
#             dobj = token

#     if all([subject, verb, dobj]):
#         # Now, how to reorder? Get the actual text, including any attached children.
#         # This part gets complex fast for Member 2, but for Member 1, just conceptualize.
        
#         # A very simplified reordering, just based on token text.
#         # Member 2 will need to handle phrases (e.g., "a red book")
#         reordered_tokens = []
#         # Get subject phrase
#         reordered_tokens.extend([t.text for t in subject.subtree])
#         # Get direct object phrase
#         reordered_tokens.extend([t.text for t in dobj.subtree])
#         # Get verb phrase (just the verb for now)
#         reordered_tokens.append(verb.text)

#         # Handle punctuation if present at the end of the sentence
#         last_token = doc[-1]
#         if last_token.is_punct:
#             reordered_tokens.append(last_token.text)
        
#         return " ".join(reordered_tokens).replace(" .", ".") # Simple cleanup

#     return None

# print("\n--- Conceptual Rule 1 Application Test ---")
# test_sentence_r1 = "She eats sushi."
# doc_r1 = nlp(test_sentence_r1)
# reordered_r1 = conceptual_apply_rule1(doc_r1)
# if reordered_r1:
#     print(f"Original: '{test_sentence_r1}'")
#     print(f"Conceptual Reordered: '{reordered_r1}'")
# else:
#     print(f"Rule 1 did not apply to '{test_sentence_r1}' conceptually.")
