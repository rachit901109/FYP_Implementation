import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
from pprint import pprint

# Import libraries for BLEU and ROUGE scores
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Load environment variables
load_dotenv()

# Define models
MODELS = {
    "llama_small": "llama3-8b-8192",  # Small Llama model
    "llama_large": "llama-3.3-70b-versatile",  # Large Llama model
    "mistral": "deepseek-r1-distill-llama-70b",  # Mistral model
    "gemini": "gemma2-9b-it",  # Gemini model (if available via Groq)
}

# Medical queries to test
MEDICAL_QUERIES = [
    "What are the primary symptoms of type 2 diabetes?",
    "Explain the process of diagnosing and treating pneumonia",
    "What are the latest treatments for rheumatoid arthritis?",
    "Describe the impact of diet on cardiovascular health",
    "How do vaccines work to prevent infectious diseases?",
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
        if all(
            failures >= self.max_failures_per_key
            for failures in self.key_failures.values()
        ):
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
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
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
                messages.append(
                    {
                        "role": "system",
                        "content": f"Use the following medical context to inform your response: {context}",
                    }
                )
            messages.append({"role": "user", "content": prompt})

            # Make API call
            completion = groq_client.chat.completions.create(
                model=model, messages=messages, max_tokens=300
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
        "What are the primary symptoms of type 2 diabetes?": "Type 2 diabetes is a chronic condition affecting metabolism. Key medical terms include insulin resistance, hyperglycemia, and metabolic syndrome. Primary symptoms include increased thirst (polydipsia), frequent urination (polyuria), increased hunger (polyphagia), fatigue, blurred vision, slow-healing sores, and frequent infections. The condition is characterized by high blood sugar levels due to insulin resistance or inadequate insulin production by the pancreas.",
        "Explain the process of diagnosing and treating pneumonia": "Pneumonia is a lung infection that can be caused by bacteria, viruses, or fungi. Diagnosis involves chest X-rays, blood tests, and sputum cultures. Physicians may use auscultation to listen for abnormal lung sounds, and may order CT scans in complex cases. Treatment varies based on pathogen; bacterial pneumonia typically requires antibiotics, viral pneumonia may need antivirals, and fungal pneumonia requires antifungal medications. Supportive care includes oxygen therapy, hydration, and respiratory support in severe cases.",
        "What are the latest treatments for rheumatoid arthritis?": "Rheumatoid arthritis is an autoimmune disorder. Modern treatments include biologics, disease-modifying antirheumatic drugs (DMARDs), and targeted synthetic DMARDs. Biologics like TNF inhibitors (adalimumab, etanercept), IL-6 inhibitors (tocilizumab), and T-cell costimulation modulators (abatacept) target specific immune system components. JAK inhibitors (tofacitinib, baricitinib, upadacitinib) are newer oral medications that block certain immune pathways. Treatment strategies now emphasize early intervention, treat-to-target approaches, and combination therapies to achieve remission.",
        "Describe the impact of diet on cardiovascular health": "Cardiovascular health is closely linked to diet. Key factors include cholesterol levels, inflammation markers, and nutrient intake affecting heart function. Mediterranean and DASH diets show strong evidence for cardiovascular benefit. Omega-3 fatty acids from fatty fish reduce inflammation and triglycerides. Plant sterols/stanols can lower LDL cholesterol. Excessive trans fats, refined carbohydrates, and sodium increase cardiovascular risk. Dietary fiber from whole grains, fruits, and vegetables supports healthy lipid profiles and blood pressure regulation. Polyphenols from berries, tea, and cocoa demonstrate antioxidant and anti-inflammatory properties beneficial for vascular health.",
        "How do vaccines work to prevent infectious diseases?": "Vaccines stimulate the immune system to recognize and fight specific pathogens. They contain antigens that trigger antibody production and immune memory. Modern vaccines use various platforms including mRNA (encoding viral proteins), viral vectors (non-replicating viruses carrying pathogen genes), subunit vaccines (specific pathogen proteins), and attenuated/inactivated whole pathogens. The immune response involves both humoral immunity (antibodies) and cell-mediated immunity (T-cells). Herd immunity occurs when sufficient population percentage is vaccinated, protecting vulnerable individuals. Adjuvants are often included to enhance immune response.",
    }
    return contexts.get(query, "")


def evaluate_responses(response1, response2, reference):
    """
    Evaluate responses using multiple metrics including BLEU and ROUGE scores

    :param response1: First response to evaluate
    :param response2: Second response to evaluate
    :param reference: Reference text (typically the context) for evaluation
    :return: Dictionary of evaluation metrics
    """
    # Context relevance using cosine similarity
    vectorizer = TfidfVectorizer()
    reference_vector = vectorizer.fit_transform([reference])
    response_vectors = vectorizer.transform([response1, response2])

    context_relevance = [
        cosine_similarity(reference_vector, response_vector)[0][0] * 100
        for response_vector in response_vectors
    ]

    # Response similarity
    response_similarity = (
        cosine_similarity(vectorizer.transform([response1, response2]))[0, 1] * 100
    )

    # Response length
    length_comparison = [len(response.split()) for response in [response1, response2]]

    # BLEU score calculation (using reference as the target)
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for response in [response1, response2]:
        try:
            # We use sentence BLEU for simplicity
            score = (
                sentence_bleu(
                    [reference.split()], response.split(), smoothing_function=smoothie
                )
                * 100
            )
            bleu_scores.append(score)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            bleu_scores.append(0)

    # ROUGE score calculation
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = []
    for response in [response1, response2]:
        try:
            scores = scorer.score(reference, response)
            # We'll use ROUGE-L F1-score
            rouge_scores.append(scores["rougeL"].fmeasure * 100)
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            rouge_scores.append(0)

    return {
        "context_relevance": context_relevance,
        "response_similarity": response_similarity,
        "response_lengths": length_comparison,
        "bleu_scores": bleu_scores,
        "rouge_scores": rouge_scores,
    }


def run_evaluation():
    """
    Run comprehensive evaluation across different scenarios

    Scenario 1: Same LLM with vs without context
    Scenario 2: Small LLM with context vs Large LLM without context
    """
    # Initialize dataframe columns
    results_data = {
        "query_id": [],
        "query_text": [],
        "scenario": [],
        "model": [],
        "has_context": [],
        "context_relevance": [],
        "bleu_score": [],
        "rouge_score": [],
        "response_similarity": [],
        "response_length": [],
    }

    # Models to use for scenario 1 (same model with/without context)
    scenario1_models = ["llama_small", "mistral", "gemini"]

    # Process each query
    for query_idx, query in enumerate(MEDICAL_QUERIES):
        # Get relevant context for the query
        context = get_medical_context(query)

        # Scenario 1: Same model with and without context
        for model_key in scenario1_models:
            try:
                model_name = MODELS[model_key]
                print(f"Processing query {query_idx + 1} with model {model_key}...")

                # Get responses for model with and without context
                no_context_response = get_llm_response(model_name, query)
                with_context_response = get_llm_response(model_name, query, context)

                # Evaluate the responses
                eval_results = evaluate_responses(
                    no_context_response, with_context_response, context
                )

                # Store results for model without context
                results_data["query_id"].append(query_idx)
                results_data["query_text"].append(query)
                results_data["scenario"].append("same_model")
                results_data["model"].append(model_key)
                results_data["has_context"].append(False)
                results_data["context_relevance"].append(
                    eval_results["context_relevance"][0]
                )
                results_data["bleu_score"].append(eval_results["bleu_scores"][0])
                results_data["rouge_score"].append(eval_results["rouge_scores"][0])
                results_data["response_similarity"].append(
                    eval_results["response_similarity"]
                )
                results_data["response_length"].append(
                    eval_results["response_lengths"][0]
                )

                # Store results for model with context
                results_data["query_id"].append(query_idx)
                results_data["query_text"].append(query)
                results_data["scenario"].append("same_model")
                results_data["model"].append(model_key)
                results_data["has_context"].append(True)
                results_data["context_relevance"].append(
                    eval_results["context_relevance"][1]
                )
                results_data["bleu_score"].append(eval_results["bleu_scores"][1])
                results_data["rouge_score"].append(eval_results["rouge_scores"][1])
                results_data["response_similarity"].append(
                    eval_results["response_similarity"]
                )
                results_data["response_length"].append(
                    eval_results["response_lengths"][1]
                )
            except Exception as e:
                print(f"Error processing model {model_key}: {e}")

        # Scenario 2: Small model with context vs Large model without context
        try:
            print(f"Processing query {query_idx + 1} for scenario 2...")
            small_model = MODELS["llama_small"]
            large_model = MODELS["llama_large"]

            # Get responses
            small_with_context = get_llm_response(small_model, query, context)
            large_no_context = get_llm_response(large_model, query)

            # Evaluate the responses
            eval_results = evaluate_responses(
                large_no_context, small_with_context, context
            )

            # Store results for large model without context
            results_data["query_id"].append(query_idx)
            results_data["query_text"].append(query)
            results_data["scenario"].append("small_vs_large")
            results_data["model"].append("llama_large")
            results_data["has_context"].append(False)
            results_data["context_relevance"].append(
                eval_results["context_relevance"][0]
            )
            results_data["bleu_score"].append(eval_results["bleu_scores"][0])
            results_data["rouge_score"].append(eval_results["rouge_scores"][0])
            results_data["response_similarity"].append(
                eval_results["response_similarity"]
            )
            results_data["response_length"].append(eval_results["response_lengths"][0])

            # Store results for small model with context
            results_data["query_id"].append(query_idx)
            results_data["query_text"].append(query)
            results_data["scenario"].append("small_vs_large")
            results_data["model"].append("llama_small")
            results_data["has_context"].append(True)
            results_data["context_relevance"].append(
                eval_results["context_relevance"][1]
            )
            results_data["bleu_score"].append(eval_results["bleu_scores"][1])
            results_data["rouge_score"].append(eval_results["rouge_scores"][1])
            results_data["response_similarity"].append(
                eval_results["response_similarity"]
            )
            results_data["response_length"].append(eval_results["response_lengths"][1])
        except Exception as e:
            print(f"Error processing scenario 2: {e}")

    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results_data)

    # Save results to CSV
    results_df.to_csv("llm_evaluation_results.csv", index=False)

    return results_df


def visualize_results(results_df):
    """
    Create comprehensive visualizations of LLM evaluation results

    :param results_df: DataFrame containing evaluation results
    """
    # Set style
    sns.set(style="whitegrid")

    # --- SCENARIO 1: Same model with vs without context ---
    plt.figure(figsize=(15, 25))

    # Get only data for scenario 1
    scenario1_df = results_df[results_df["scenario"] == "same_model"].copy()

    # Metrics to visualize
    metrics = [
        {"col": "context_relevance", "title": "Context Relevance Score (%)"},
        {"col": "bleu_score", "title": "BLEU Score (%)"},
        {"col": "rouge_score", "title": "ROUGE-L Score (%)"},
    ]

    # Plot each metric
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)

        # Group the data instead of using pivot_table and melt
        # This approach is simpler and avoids the column naming issues
        g = sns.barplot(
            data=scenario1_df,
            x="query_id",
            y=metric["col"],
            hue=scenario1_df.apply(
                lambda x: f"{x['model']}"
                + (" w/ context" if x["has_context"] else " w/o context"),
                axis=1,
            ),
            palette="viridis",
        )

        # Add labels
        plt.title(
            f"Scenario 1: {metric['title']} - Same Model With vs Without Context",
            fontsize=14,
        )
        plt.xlabel("Query ID", fontsize=12)
        plt.ylabel(metric["title"], fontsize=12)
        plt.xticks(
            np.arange(len(MEDICAL_QUERIES)),
            [f"Q{i + 1}" for i in range(len(MEDICAL_QUERIES))],
        )
        plt.legend(title="Model & Context", fontsize=10)

    plt.tight_layout()
    plt.savefig("scenario1_evaluation.png", dpi=300)
    plt.close()

    # --- SCENARIO 2: Small model with context vs Large model without context ---
    plt.figure(figsize=(15, 25))

    # Get only data for scenario 2
    scenario2_df = results_df[results_df["scenario"] == "small_vs_large"].copy()

    # Plot each metric
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)

        # Use direct plotting instead of pivot and melt
        g = sns.barplot(
            data=scenario2_df,
            x="query_id",
            y=metric["col"],
            hue=scenario2_df.apply(
                lambda x: f"{x['model']}"
                + (" w/ context" if x["has_context"] else " w/o context"),
                axis=1,
            ),
            palette="magma",
        )

        # Add labels
        plt.title(
            f"Scenario 2: {metric['title']} - Small w/ Context vs Large w/o Context",
            fontsize=14,
        )
        plt.xlabel("Query ID", fontsize=12)
        plt.ylabel(metric["title"], fontsize=12)
        plt.xticks(
            np.arange(len(MEDICAL_QUERIES)),
            [f"Q{i + 1}" for i in range(len(MEDICAL_QUERIES))],
        )
        plt.legend(title="Model & Context", fontsize=10)

        # Add labels
        plt.title(
            f"Scenario 2: {metric['title']} - Small w/ Context vs Large w/o Context",
            fontsize=14,
        )
        plt.xlabel("Query ID", fontsize=12)
        plt.ylabel(metric["title"], fontsize=12)
        plt.xticks(
            np.arange(len(MEDICAL_QUERIES)),
            [f"Q{i + 1}" for i in range(len(MEDICAL_QUERIES))],
        )
        plt.legend(title="Model & Context", fontsize=10)

    plt.tight_layout()
    plt.savefig("scenario2_evaluation.png", dpi=300)
    plt.close()

    # --- OVERALL PERFORMANCE HEATMAP ---
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap by manually creating the matrix
    # Group by model and context first
    grouped_data = (
        results_df.groupby(["model", "has_context"])
        .agg({"context_relevance": "mean", "bleu_score": "mean", "rouge_score": "mean"})
        .reset_index()
    )

    # Create labels for each model+context combination
    model_contexts = []
    for _, row in grouped_data.iterrows():
        model_contexts.append(
            f"{row['model']}"
            + (" w/ context" if row["has_context"] else " w/o context")
        )

    # Prepare data for the heatmap
    heatmap_data = np.zeros((len(model_contexts), 3))
    for i, (_, row) in enumerate(grouped_data.iterrows()):
        heatmap_data[i, 0] = row["context_relevance"]
        heatmap_data[i, 1] = row["bleu_score"]
        heatmap_data[i, 2] = row["rouge_score"]

    # Create DataFrame for heatmap
    heatmap_matrix = pd.DataFrame(
        heatmap_data,
        index=model_contexts,
        columns=["Context Relevance", "BLEU Score", "ROUGE-L Score"],
    )

    # Rename columns for better display
    heatmap_matrix.columns = ["Context Relevance", "BLEU Score", "ROUGE-L Score"]

    # Create the heatmap
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)

    plt.title("Overall Performance Comparison Across Models and Contexts", fontsize=16)
    plt.tight_layout()
    plt.savefig("overall_performance_heatmap.png", dpi=300)
    plt.close()

    # --- RADAR CHART FOR MODEL COMPARISON ---
    plt.figure(figsize=(10, 10))

    # Prepare data
    categories = ["Context Relevance", "BLEU Score", "ROUGE-L Score"]
    N = len(categories)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialize the radar plot
    ax = plt.subplot(111, polar=True)

    # Set the labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw y-axis labels (concentric circles)
    ax.set_rlabel_position(0)
    plt.yticks(
        [20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10
    )
    plt.ylim(0, 100)

    # Plot each model using the heatmap_matrix we created earlier
    for i, model_context in enumerate(heatmap_matrix.index):
        values = heatmap_matrix.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model_context)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Comparison Across Metrics", size=15)

    plt.tight_layout()
    plt.savefig("model_radar_chart.png", dpi=300)
    plt.close()

    # Return the file paths
    return [
        "scenario1_evaluation.png",
        "scenario2_evaluation.png",
        "overall_performance_heatmap.png",
        "model_radar_chart.png",
    ]


if __name__ == "__main__":
    # Run evaluation
    print("Starting LLM response evaluation...")
    results = run_evaluation()

    # Visualize results
    print("Creating visualizations...")
    visualization_files = visualize_results(results)

    print(f"Evaluation complete. Results saved to CSV and visualizations created:")
    for viz_file in visualization_files:
        print(f"- {viz_file}")
