import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

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
    # Skip evaluation if either response is empty
    if not response1 or not response2:
        return {
            "context_relevance": [0, 0],
            "response_similarity": 0,
            "response_lengths": [0, 0]
        }
    
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
        "scenarios": [],
        "context_relevance": [],
        "response_similarity": [],
        "length_diff": []
    }
    
    for query in MEDICAL_QUERIES:
        # Get context
        context = get_medical_context(query)
        
        # Scenario 1: Small model without context vs Small model with context
        small_no_context = get_llm_response(MODELS["small"], query)
        small_with_context = get_llm_response(MODELS["small"], query, context)
        eval1 = evaluate_responses(small_no_context, small_with_context, context)
        
        results["scenarios"].append("Small: No Context vs With Context")
        results["context_relevance"].append(eval1["context_relevance"])
        results["response_similarity"].append(eval1["response_similarity"])
        results["length_diff"].append(abs(eval1["response_lengths"][0] - eval1["response_lengths"][1]))
        
        # Scenario 2: Large model without context vs Small model with context
        large_no_context = get_llm_response(MODELS["large"], query)
        small_with_context = get_llm_response(MODELS["small"], query, context)
        eval2 = evaluate_responses(large_no_context, small_with_context, context)
        
        results["scenarios"].append("Large vs Small with Context")
        results["context_relevance"].append(eval2["context_relevance"])
        results["response_similarity"].append(eval2["response_similarity"])
        results["length_diff"].append(abs(eval2["response_lengths"][0] - eval2["response_lengths"][1]))
    
    return results

def plot_context_relevance(results):
    """
    Create a detailed plot for Context Relevance for two distinct scenarios
    """
    plt.figure(figsize=(15, 7))
    plt.title('Context Relevance Across Different Scenarios', fontsize=15)
    
    # Separate the two scenarios' context relevance
    scenario1_relevance = [float(val) for val in results['context_relevance'][::2]]
    scenario2_relevance = [float(val) for val in results['context_relevance'][1::2]]
    scenarios1 = results['scenarios'][::2]
    scenarios2 = results['scenarios'][1::2]
    
    # Create subplot for first set of scenarios
    plt.subplot(1, 2, 1)
    plt.bar(scenarios1, scenario1_relevance)
    plt.title('Scenario Set 1: Context Relevance', fontsize=12)
    plt.xlabel('Scenarios', fontsize=10)
    plt.ylabel('Context Relevance (%)', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
    # Create subplot for second set of scenarios
    plt.subplot(1, 2, 2)
    plt.bar(scenarios2, scenario2_relevance)
    plt.title('Scenario Set 2: Context Relevance', fontsize=12)
    plt.xlabel('Scenarios', fontsize=10)
    plt.ylabel('Context Relevance (%)', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('context_relevance.png')
    plt.close()

def plot_response_similarity(results):
    """
    Create a bar plot for Response Similarity
    """
    plt.figure(figsize=(12, 6))
    plt.title('Response Similarity Across Different Scenarios', fontsize=15)
    
    scenarios = results['scenarios']
    similarities = results['response_similarity']
    
    plt.bar(scenarios, similarities)
    plt.xlabel('Evaluation Scenarios', fontsize=12)
    plt.ylabel('Response Similarity (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('response_similarity.png')
    plt.close()

def plot_length_difference(results):
    """
    Create a bar plot for Response Length Difference
    """
    plt.figure(figsize=(12, 6))
    plt.title('Response Length Difference Across Scenarios', fontsize=15)
    
    scenarios = results['scenarios']
    length_diffs = results['length_diff']
    
    plt.bar(scenarios, length_diffs)
    plt.xlabel('Evaluation Scenarios', fontsize=12)
    plt.ylabel('Word Count Difference', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('length_difference.png')
    plt.close()

def plot_performance_heatmap(results):
    """
    Create a comprehensive performance heatmap
    """
    plt.figure(figsize=(15, 10))
    
    # Prepare data for heatmap
    # Flatten context_relevance but separate into two sets
    context_relevance_1 = [float(val) for val in results['context_relevance'][::2]]
    context_relevance_2 = [float(val) for val in results['context_relevance'][1::2]]
    
    # Prepare performance data
    performance_data_1 = np.column_stack([
        context_relevance_1,
        [results['response_similarity'][0]] * len(context_relevance_1),
        [results['length_diff'][0]] * len(context_relevance_1)
    ])
    
    performance_data_2 = np.column_stack([
        context_relevance_2,
        [results['response_similarity'][1]] * len(context_relevance_2),
        [results['length_diff'][1]] * len(context_relevance_2)
    ])
    
    # Combine performance data
    performance_data = np.vstack([performance_data_1, performance_data_2])
    
    plt.title('Comparative Performance Heatmap', fontsize=15)
    sns.heatmap(performance_data, 
                cmap='YlGnBu', 
                xticklabels=['Context Relevance', 'Response Similarity', 'Length Diff'],
                yticklabels=results['scenarios'],
                annot=True,
                fmt='.2f')
    plt.xlabel('Performance Metrics', fontsize=12)
    plt.ylabel('Evaluation Scenarios', fontsize=12)
    plt.tight_layout()
    plt.savefig('performance_heatmap.png')
    plt.close()

# Run the full evaluation
evaluation_results = run_evaluation()

# Generate individual plots
plot_context_relevance(evaluation_results)
plot_response_similarity(evaluation_results)
plot_length_difference(evaluation_results)
plot_performance_heatmap(evaluation_results)

print("Evaluation complete. Results visualizations saved.")
