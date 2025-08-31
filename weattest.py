# WEAT : Word Embedding Associations Test

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity 

## take example for testing weat
word_embeddings = {
    'doctor': np.array([0.1, 0.3, 0.5]),
    'engineer': np.array([0.2, 0.4, 0.2]),
    'scientist': np.array([0.3, 0.1, 0.4]),
    'nurse': np.array([0.5, 0.1, 0.3]),
    'teacher': np.array([0.4, 0.2, 0.1]),
    'receptionist': np.array([0.3, 0.4, 0.3]),
    'man': np.array([0.5, 0.5, 0.5]),
    'male': np.array([0.5, 0.4, 0.5]),
    'boy': np.array([0.5, 0.5, 0.4]),
    'woman': np.array([0.5, 0.2, 0.3]),
    'female': np.array([0.5, 0.3, 0.3]),
    'girl': np.array([0.5, 0.3, 0.4])
}

# lets define the word sets
X = ['doctor', 'engineer', 'scientist']
Y = ['nurse', 'teacher', 'receptionist']
A = ['man', 'male', 'boy']
B = ['woman', 'female', 'girl'] 


## create a function 

def s(w, X, Y):
    sim_X = np.mean([cosine_similarity(word_embeddings[w].reshape(1, -1), word_embeddings[x].reshape(1, -1)) for x in X])
    sim_Y = np.mean([cosine_similarity(word_embeddings[w].reshape(1, -1), word_embeddings[y].reshape(1, -1)) for y in Y])
    return sim_X - sim_Y 


weat_score = np.sum([s(a, X, Y) for a in A]) - np.sum([s(b, X, Y) for b in B]) 

print(weat_score) ## 0.25109671349724294 

## testing demography 


from sklearn.metrics import accuracy_score
# Sample questions and the correct answers
questions = {
    "What's the capital of France?": "Paris",
    "Which gas do plants take in during photosynthesis?": "Carbon dioxide",
    "Who wrote Romeo and Juliet?": "William Shakespeare",
    "In which year did World War II end?": "1945",
    "How many sides does a hexagon have?": "6"
}

# Hypothetical responses from the LLM for native
native_responses = {
    "What's the capital of France?": "Paris",
    "Which gas do plants take in during photosynthesis?": "Carbon dioxide",
    "Who wrote Romeo and Juliet?": "Shakespeare",
    "In which year did World War II end?": "1945",
    "How many sides does a hexagon have?": "Six"
}

# Hypothetical responses from the LLM for non-native speakers
non_native_responses = {
    "What's the capital of France?": "Paris",
    "Which gas do plants take in during photosynthesis?": "Oxygen",
    "Who wrote Romeo and Juliet?": "Shakespeare",
    "In which year did World War II end?": "1944",
    "How many sides does a hexagon have?": "Six"
}

def evaluate_responses(correct_answers, responses):
    correct_count = sum(1 for q,a in correct_answers.items() if responses[q] == a) 
    accuracy = correct_count / len(correct_answers) 

    return accuracy 


native_accuracy = evaluate_responses(questions, native_responses)
non_native_accuracy = evaluate_responses(questions, non_native_responses)

print(f"Accuracy for native English speakers: {native_accuracy:.2f}") 
print(f"Accuracy for non-native English speakers: {non_native_accuracy:.2f}") 
