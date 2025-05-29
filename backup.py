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

