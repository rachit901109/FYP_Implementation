import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
from pprint import pprint

# Load environment variables
load_dotenv()

# Define models
MODELS = {
    "small": "llama3-8b-8192",
    "large": "llama-3.3-70b-versatile"
}

# Medical queries to test
MEDICAL_QUERIES = [
    "What are the primary symptoms of type 2 diabetes?",
    "Explain the process of diagnosing and treating pneumonia",
    "What are the latest treatments for rheumatoid arthritis?",
    "Describe the impact of diet on cardiovascular health",
    "How do vaccines work to prevent infectious diseases?"
]

class APIKeyRotator:
    def __init__(self, api_keys):
        """
        Initialize API key rotator
        
        :param api_keys: List of API keys to rotate through
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_failures = {key: 0 for key in api_keys}
        self.max_failures_per_key = 3
    
    def get_next_key(self):
        """
        Get the next available API key
        
        :return: Next API key to use
        """
        # Shuffle keys if all keys have reached max failures
        if all(failures >= self.max_failures_per_key for failures in self.key_failures.values()):
            self.key_failures = {key: 0 for key in self.api_keys}
        
        # Rotate through keys
        original_index = self.current_key_index
        while True:
            current_key = self.api_keys[self.current_key_index]
            
            # Move to next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            # If key hasn't reached max failures, return it
            if self.key_failures[current_key] < self.max_failures_per_key:
                return current_key
            
            # Prevent infinite loop
            if self.current_key_index == original_index:
                raise ValueError("All API keys have reached maximum failure limit")

def get_llm_response(model, prompt, context=None, max_retries=3):
    """
    Get response from Groq LLM with optional context and API key rotation
    
    :param model: Groq model to use
    :param prompt: User prompt
    :param context: Optional context for the prompt
    :param max_retries: Maximum retry attempts
    :return: LLM response or empty string
    """
    # Collect API keys from environment variables
    api_keys = [
        os.environ.get('GROQ_API_KEY_1'),
        os.environ.get('GROQ_API_KEY_2'),
        os.environ.get('GROQ_API_KEY_3')
    ]
    api_keys = [key for key in api_keys if key]  # Remove any None values
    
    if not api_keys:
        raise ValueError("No API keys found in environment variables")
    
    # Initialize API key rotator
    key_rotator = APIKeyRotator(api_keys)
    
    for attempt in range(max_retries):
        try:
            # Get next available API key
            current_api_key = key_rotator.get_next_key()
            
            # Create Groq client with current API key
            groq_client = Groq(api_key=current_api_key)
            
            # Prepare messages
            messages = []
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Use the following medical context to inform your response: {context}"
                })
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            completion = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error getting response (Attempt {attempt + 1}): {e}")
            
            # Mark the current key as failed
            key_rotator.key_failures[current_api_key] += 1
            
            # Wait before retrying
            time.sleep(2)
    
    # If all attempts fail
    print("Failed to get response after all retry attempts")
    return ""

def get_medical_context(query):
    """
    Simulate getting medical context (in a real scenario, this would be a more sophisticated retrieval)
    """
    contexts = {
        "What are the primary symptoms of type 2 diabetes?": "Type 2 diabetes is a chronic condition affecting metabolism. Key medical terms include insulin resistance, hyperglycemia, and metabolic syndrome.",
        "Explain the process of diagnosing and treating pneumonia": "Pneumonia is a lung infection that can be caused by bacteria, viruses, or fungi. Diagnosis involves chest X-rays, blood tests, and sputum cultures.",
        "What are the latest treatments for rheumatoid arthritis?": "Rheumatoid arthritis is an autoimmune disorder. Modern treatments include biologics, disease-modifying antirheumatic drugs (DMARDs), and targeted synthetic DMARDs.",
        "Describe the impact of diet on cardiovascular health": "Cardiovascular health is closely linked to diet. Key factors include cholesterol levels, inflammation markers, and nutrient intake affecting heart function.",
        "How do vaccines work to prevent infectious diseases?": "Vaccines stimulate the immune system to recognize and fight specific pathogens. They contain antigens that trigger antibody production and immune memory."
    }
    return contexts.get(query, "")


def evaluate_responses(response1, response2, context):
    """
    Evaluate responses using multiple metrics
    """
    # Context relevance using cosine similarity
    vectorizer = TfidfVectorizer()
    context_vector = vectorizer.fit_transform([context])
    response_vectors = vectorizer.transform([response1, response2])
    
    context_relevance = [
        cosine_similarity(context_vector, response_vector)[0][0] * 100 
        for response_vector in response_vectors
    ]
    
    # Response similarity
    response_similarity = cosine_similarity(
        vectorizer.transform([response1, response2])
    )[0, 1] * 100
    
    # Response length
    length_comparison = [len(response.split()) for response in [response1, response2]]
    
    return {
        "context_relevance": context_relevance,
        "response_similarity": response_similarity,
        "response_lengths": length_comparison
    }

def run_evaluation():
    """
    Run comprehensive evaluation across different scenarios
    """
    results = {
        "context_relevance": [],
        "response_similarity": [],
        "length_diff": []
    }
    
    for query in MEDICAL_QUERIES:
        # Get context
        context = get_medical_context(query)
        
        # Get responses
        # Scenario 1: Small model without context vs Small model with context
        small_no_context = get_llm_response(MODELS["small"], query)
        small_with_context = get_llm_response(MODELS["small"], query, context)
        
        # Scenario 2: Large model without context vs Small model with context
        large_no_context = get_llm_response(MODELS["large"], query)
        
        # Evaluate Scenario 1
        eval1 = evaluate_responses(small_no_context, small_with_context, context)
        results["context_relevance"].append(eval1["context_relevance"])
        results["response_similarity"].append(eval1["response_similarity"])
        results["length_diff"].append(abs(eval1["response_lengths"][0] - eval1["response_lengths"][1]))
        
        # Evaluate Scenario 2
        eval2 = evaluate_responses(large_no_context, small_with_context, context)
        results["context_relevance"].append(eval2["context_relevance"])
        results["response_similarity"].append(eval2["response_similarity"])
        results["length_diff"].append(abs(eval2["response_lengths"][0] - eval2["response_lengths"][1]))
    
    return results

#def visualize_results(results):
#    """
#    Create comprehensive visualization of evaluation results
#    """
#    plt.figure(figsize=(15, 10))
#    
#    # Context Relevance
#    plt.subplot(2, 2, 1)
#    sns.boxplot(data=results['context_relevance'])
#    plt.title('Context Relevance Across Queries')
#    plt.xlabel('Query')
#    plt.ylabel('Context Relevance (%)')
#    
#    # Response Similarity
#    plt.subplot(2, 2, 2)
#    plt.bar(range(len(results['response_similarity'])), results['response_similarity'])
#    plt.title('Response Similarity')
#    plt.xlabel('Query Pair')
#    plt.ylabel('Similarity (%)')
#    
#    # Length Difference
#    plt.subplot(2, 2, 3)
#    plt.bar(range(len(results['length_diff'])), results['length_diff'])
#    plt.title('Response Length Difference')
#    plt.xlabel('Query Pair')
#    plt.ylabel('Word Count Difference')
#    
#    # Heatmap of Comparative Performance
#    plt.subplot(2, 2, 4)
#    performance_matrix = np.column_stack([
#        results['context_relevance'],
#        results['response_similarity'],
#        results['length_diff']
#    ])
#    sns.heatmap(performance_matrix, cmap='YlGnBu', 
#                xticklabels=['Context Relevance', 'Response Similarity', 'Length Diff'],
#                yticklabels=[f'Query {i+1}' for i in range(len(performance_matrix))])
#    plt.title('Comparative Performance Heatmap')
#    
#    plt.tight_layout()
#    plt.savefig('hybrid_rag_evaluation.png')
#    plt.close()
#

#def visualize_results(results):
#    """
#    Create separate visualizations for evaluation metrics with clear scenario labels
#    """
#    sns.set(style="whitegrid")
#    
#    # Define scenarios for better labeling
#    scenario_labels = ["Small w/o Context", "Small w/ Context",
#                       "Large w/o Context", "Small w/ Context"]
#    
#    num_queries = len(results["context_relevance"]) // 2  # Each query has two comparisons
#    
#    # --- Context Relevance ---
#    plt.figure(figsize=(10, 5))
#    context_relevance_flat = np.array(results['context_relevance']).flatten()
#    sns.barplot(x=np.tile(scenario_labels, num_queries), y=context_relevance_flat, palette="Blues")
#    plt.title("Context Relevance Across Scenarios")
#    plt.xlabel("Model Scenario")
#    plt.ylabel("Context Relevance (%)")
#    plt.xticks(rotation=30)
#    plt.savefig("context_relevance.png")
#    plt.show()
#    
#    # --- Response Similarity ---
#    plt.figure(figsize=(10, 5))
#    sns.barplot(x=scenario_labels[:2] * num_queries, y=results['response_similarity'], palette="Greens")
#    plt.title("Response Similarity Across Scenarios")
#    plt.xlabel("Model Scenario")
#    plt.ylabel("Similarity (%)")
#    plt.xticks(rotation=30)
#    plt.savefig("response_similarity.png")
#    plt.show()
#    
#    # --- Length Difference ---
#    plt.figure(figsize=(10, 5))
#    sns.barplot(x=scenario_labels[:2] * num_queries, y=results['length_diff'], palette="Oranges")
#    plt.title("Response Length Difference Across Scenarios")
#    plt.xlabel("Model Scenario")
#    plt.ylabel("Word Count Difference")
#    plt.xticks(rotation=30)
#    plt.savefig("length_difference.png")
#    plt.show()
#
#    # --- Heatmap of Comparative Performance ---
#    plt.figure(figsize=(10, 6))
#    performance_matrix = np.column_stack([
#        results['context_relevance'],
#        results['response_similarity'],
#        results['length_diff']
#    ])
#    sns.heatmap(performance_matrix, cmap="coolwarm", annot=True, fmt=".2f",
#                xticklabels=['Context Relevance', 'Response Similarity', 'Length Diff'],
#                yticklabels=[f'Query {i+1}' for i in range(num_queries * 2)])
#    plt.title("Comparative Performance Heatmap")
#    plt.savefig("performance_heatmap.png")
#    plt.show()
#

def visualize_results(results, queries, scenarios):
    """
    Create separate visualizations for evaluation metrics for each medical query across scenarios.
    """
    sns.set(style="whitegrid")

    # Convert data into a DataFrame for Seaborn
    data_list = []
    for query_idx, query in enumerate(queries):
        for scenario_idx, scenario in enumerate(scenarios):
            data_list.append([query, scenario, results['context_relevance'][query_idx][scenario_idx], 
                              results['response_similarity'][query_idx][scenario_idx], 
                              results['length_diff'][query_idx][scenario_idx]])

    # Create DataFrame
    df = pd.DataFrame(data_list, columns=["Query", "Scenario", "Context Relevance", "Response Similarity", "Length Difference"])

    # --- Context Relevance ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Query", y="Context Relevance", hue="Scenario", palette="Blues")
    plt.title("Context Relevance Across Queries and Scenarios")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scenario")
    plt.savefig("context_relevance.png")
    plt.show()

    # --- Response Similarity ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Query", y="Response Similarity", hue="Scenario", palette="Greens")
    plt.title("Response Similarity Across Queries and Scenarios")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scenario")
    plt.savefig("response_similarity.png")
    plt.show()

    # --- Length Difference ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Query", y="Length Difference", hue="Scenario", palette="Oranges")
    plt.title("Response Length Difference Across Queries and Scenarios")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scenario")
    plt.savefig("length_difference.png")
    plt.show()

    # --- Heatmap of Comparative Performance ---
    heatmap_data = df.pivot(index="Query", columns="Scenario", values="Context Relevance")
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Comparative Performance Heatmap")
    plt.savefig("performance_heatmap.png")
    plt.show()

# Run the full evaluation
evaluation_results = run_evaluation()
scenarios = ["Small Model (No Context)", "Small Model (With Context)",
             "Large Model (No Context)"]
visualize_results(evaluation_results, MEDICAL_QUERIES, scenarios)

print("Evaluation complete. Results visualization saved as 'hybrid_rag_evaluation.png'")
pprint(evaluation_results, sort_dicts=False)
