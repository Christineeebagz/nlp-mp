import spacy

def identify_svo(sentence):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the sentence
    doc = nlp(sentence)
    
    # Initialize variables
    subject = None
    verb = None
    obj = None
    
    # Iterate through tokens to find SVO
    for token in doc:
        # Find subject (usually a noun with nsubj dependency)
        if "subj" in token.dep_:
            subject = token.text
            # Include compound nouns (e.g., "dog food" as subject)
            for child in token.children:
                if child.dep_ == "compound":
                    subject = f"{child.text} {token.text}"
        
        # Find main verb (usually ROOT of the sentence)
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.text
        
        # Find object (usually a noun with dobj dependency)
        if "dobj" in token.dep_:
            obj = token.text
            # Include compound nouns for objects too
            for child in token.children:
                if child.dep_ == "compound":
                    obj = f"{child.text} {token.text}"
    
    return {
        "sentence": sentence,
        "subject": subject,
        "verb": verb,
        "object": obj
    }

# Example usage
sentence = input("Enter sentence: ")
result = identify_svo(sentence)

# Print in Subject-Object-Verb (SOV) order
print("\nAnalysis:")
print(f"Sentence: {result['sentence']}")
print(f"Subject: {result['subject']}")
print(f"Object: {result['object']}")
print(f"Verb: {result['verb']}")

# Bonus: Print reconstructed sentence in SOV order (if all components found)
if result['subject'] and result['object'] and result['verb']:
    print("\nReconstructed (SOV order):")
    print(f"{result['subject']} {result['object']} {result['verb']}")
else:
    print("\nNote: Could not reconstruct full SOV sentence (missing components)")


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