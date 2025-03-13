from spoke import get_wikipedia_info

# Function to extract medical entities using SpaCy
def extract_medical_entities(text):
    keywords = []
    with open('keywords.txt', 'r') as f:
        medical_keywords = [line.rstrip('\n') for line in f]
    
    tokens = text.replace('.', '').split(' ')
    for token in tokens:
        print(token.lower())
        if token.lower() in medical_keywords:
            keywords.append(token)

    print("found keywords:-")
    print(keywords)
    
    return keywords 

# Function to fetch medical context information
def get_medical_context(entities):
    context = "### Context for Medical Terms\n"
    for term in entities:
        context += f"- {term}:{get_wikipedia_info(term)}" 
        context+="\n\n"
    return context

user_input = 'Give me detail information and difference between Dermatofibroma and Keratosis.'

medical_entities = extract_medical_entities(user_input)
medical_context = get_medical_context(medical_entities)
print(medical_context)