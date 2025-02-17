# --- Import ---
import streamlit as st
import numpy as np
import os
import platform
from PIL import Image
from keras._tf_keras.keras.models import load_model
from spoke import SpokeNetwork, get_search, get_wikipedia_info
from utils import save_file, preprocess_image, MODEL_PATH
import pandas as pd
from datetime import datetime
import json
import requests
from langchain_community.retrievers import WikipediaRetriever
import streamlit.components.v1 as components
from groq import Groq
import tiktoken
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

# --- Cache model for performacne ---
@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)
model = load_prediction_model()

# --- Diseases in HAM10000 dataset ---
# {'bkl': 2, 'nv': 4, 'df': 3, 'mel': 6, 'vasc': 5, 'bcc': 1, 'akiec': 0}
class_names = {
    2: 'Benign Keratosis',
    4: 'Melanocytic Nevi',
    3: 'Dermatofibroma',
    6: 'Melanoma',
    5: 'Vascular Lesions',
    1: 'Basal Cell Carcinoma',
    0: 'Actinic Keratosis'
}

api_key = os.environ.get('groq_api_key')
client = Groq(api_key=api_key)

# --- Helper functions for model, llm analysis ---
def predict_single_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction_prob = model.predict(img_array)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    return class_names[predicted_class],f'{prediction_prob[0][predicted_class]*100:.2f}'

# relationship extraction for LLM analysis
def extract_disease_relationships(data):
    """
    Converts a neighborhood graph for a disease into text to be processed by an LLM.
    Example: (d1) --TREATS_CtD--> (d2) -> "d1 treats d2"
    """
    edge_mapping = {
        "ISA_DiD": "{} is a type of {}.",
        "TREATS_CtD": "{} treats {}.",
        "ASSOCIATES_DaG": "{} is associated with gene {}.",
        "PRESENTS_DpS": "{} presents the symptom {}.",
        "CONTAINS_DScB": "{} contains {}.",
        "CAUSES_OcD": "{} causes {}.",
        "LOCALIZES_DlA": "{} localizes to {}.",
        "PREVALENCE_DpL": "{} has prevalence in {}.",
        "RESEMBLES_DrD": "{} resembles {}.",
        "MARKER_NEG_GmnD": "{} is a negative marker for {}.",
        "MARKER_POS_GmpD": "{} is a positive marker for {}.",
        "CONTRAINDICATES_CcD": "{} is contraindicated for {}.",
        "INCREASEDIN_PiD": "{} increases in {}.",
        "DECREASEDIN_PdD": "{} decreases in {}.",
        "BINDS_CbP": "{} binds to protein {}.",
        "INTERACTS_PiP": "{} interacts with {}.",
        "INTERACTS_PiC": "{} interacts with compound {}.",
        "REGULATES_PrG": "{} regulates gene {}.",
        "DOWNREGULATES_CdG": "{} downregulates gene {}.",
        "UPREGULATES_CuG": "{} upregulates gene {}.",
        "PARTICIPATES_GpBP": "{} participates in biological process {}.",
        "PARTICIPATES_GpMF": "{} participates in molecular function {}.",
        "PARTICIPATES_GpPW": "{} participates in pathway {}.",
        "PARTICIPATES_GpR": "{} participates in reaction {}.",
        "PRODUCES_RpC": "{} produces compound {}.",
        "CONSUMES_RcC": "{} consumes compound {}.",
        "FOUNDIN_CfL": "{} is found in location {}.",
        "ASSOCIATES_SaD": "{} is associated with disease {}.",
        "TARGETS_MtG": "{} targets gene {}.",
        "RESISTANT_TO_mGrC": "{} is resistant to {}.",
        "TREATS_CtD": "{} (compound) treats disease {}.",
        "RESPONSE_TO_mGrC": "{} responds to {}."
    }

    relationships = []
    node_lookup = {node["data"]["id"]: node["data"]["properties"].get("name", "Unknown") for node in data}

    for entry in data:
        edge_type = entry["data"].get("neo4j_type")
        if edge_type in edge_mapping:
            source_id = entry["data"]["source"]
            target_id = entry["data"]["target"]
            source_name = node_lookup.get(source_id, "Unknown Source")
            target_name = node_lookup.get(target_id, "Unknown Target")

            relationships.append(edge_mapping[edge_type].format(source_name, target_name))

    return relationships

def truncate_text(text, max_tokens=5000):
    """Truncates the text to fit within the token limit."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def get_groq_output(disease, information):
    system_prompt = (
        "You are an advanced biomedical AI specializing in dermatology and skin diseases trained on the HAM10000 dataset. "
        "Your role is to provide comprehensive, evidence-based, and well-structured medical analysis of skin diseases "
        "for patients, doctors, and researchers. Your responses should be accurate and detail"
        "You must describe its general definition, causes, symptoms, and available treatments. "
    )
    
    information_text = "\n".join(information)
    truncated_info = truncate_text(information_text, max_tokens=5000)
    
    user_prompt = (
        f"I am researching the skin disease **{disease}**. "
        "Please provide a detailed medical explanation including:\n\n"
        "1. **General Overview**: What is this disease?\n"
        "2. **Causes**: What are the possible causes?\n"
        "3. **Symptoms**: What symptoms are commonly associated with this disease?\n"
        "4. **Treatment Options**: What are the available treatments?\n\n"
        "Here is some extracted information from a biomedical database (SPOKE) related to this disease:\n"
        f"{truncated_info}\n\n"
        "Use the above data along with your medical knowledge to generate a structured response."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# --- Streamlit application ---
def main():
    st.title("Hybrid RAG Classifier App")
    # FUTURE SCOPE:- pdf and txt inputs for medical reports, currently upload images for diagnosis
    uploaded_file = st.file_uploader("Upload an Image, PDF, or Text file", type=["jpg", "jpeg", "png", "pdf", "txt"])

    if uploaded_file is not None:
        file_path = save_file(uploaded_file)
        st.success(f"File successfully uploaded and saved to: {file_path}")

        # Handle different file types
        if uploaded_file.type.startswith("image"):
            st.image(file_path, caption="Uploaded Image", use_container_width=True)
            predicted_class,confidence = predict_single_image(model, file_path)
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence}")

            # retrieve context for predicted class
            with st.spinner('Collecting context from knowledge base...'):
                # Initialize SPOKE network
                network = SpokeNetwork()
                search_results = get_search(predicted_class)
                disease_entries = {
                    item['name']:item for item in search_results if item['node_type'] == "Disease"
                }
                neighborhood = { dis['identifier']:network.fetch_neighborhood(
                        "Disease", 
                        "identifier", 
                        dis['identifier']
                    ) for name, dis in disease_entries.items()
                }
                wiki_info = get_wikipedia_info(predicted_class)
                print("Search Results:-")
                pprint(search_results, sort_dicts=False)
                print("--"*30)
                print("Disease Entries:-")
                pprint(disease_entries, sort_dicts=False)
                print("--"*30)
                print("Neighborhood Data:-")
                pprint(neighborhood, sort_dicts=False)
                print("--"*30)
                print("Wikipedia Info:-")
                pprint(wiki_info, sort_dicts=False)
                print("--"*30)

            # display llm output and context provided in tabs
            analysis_tab, wiki_tab, spoke_tab = st.tabs([
                "Disease Analysis by LLM", 
                "External Information", 
                "Disease Graph Network"
            ])

            with analysis_tab:
                st.markdown("### Comprehensive Disease Analysis")
                information = ""
                all_rel = {}
                if disease_entries:
                    formatted_data = []
                    for name, dis in disease_entries.items():
                        neighborhood_data = neighborhood.get(dis['identifier'], [])
                        relationships = extract_disease_relationships(neighborhood_data)
                        all_rel[name] = relationships
                        disease_info = {
                            "Disease": dis["name"],
                            "Identifier": dis["identifier"],
                            "Number_of_neighbors": len(neighborhood_data),
                            "Relationships": relationships if relationships else ["No significant relationships found."]
                        }
                        formatted_data.append(disease_info)

                    information = json.dumps(formatted_data, indent=4)
                else:
                    st.warning("No external disease information found in SPOKE graph.")
                print(f"relationships from spoke graph")
                pprint(all_rel, sort_dicts=False)
                print("--"*30)
                print("information given to llm from spoke graph relationship")
                print(information)
                print("--"*30)

                with st.spinner('Generating comprehensive analysis...'):
                    llm_analysis = get_groq_output(predicted_class, information)
                    st.markdown(llm_analysis)

            with wiki_tab:
                st.markdown("### Disease information from external sources")
                st.markdown(wiki_info)

            with spoke_tab:
                st.markdown("### Disease Graph Network Analysis")
                # Display search results in a table
                if disease_entries:
                    st.markdown("#### Search Results:")
                    results_df = pd.DataFrame([
                        {
                            'Name': item.get('name', ''),
                            'Node Type': item.get('node_type', ''),
                            'Identifier': item.get('identifier', '')
                        }
                        for item in search_results
                    ])
                    st.dataframe(results_df)

                    # Selectbox for diseases in search result to show corroponding network graph
                    options = [dis.get('name', "no name key") for name, dis in disease_entries.items()]
                    options.insert(0, "")

                    selected = st.selectbox(
                        "Select a related disease to view neighborhood graph:",
                        options=options,
                    )

                    dis = disease_entries.get(selected, "")
                    # Create and display network visualization
                    if selected!="":
                        try:
                            neighborhood_data = neighborhood.get(dis['identifier'], [])
                            html_file = network.create_network_visualization(
                                neighborhood_data,
                                dis['name']
                            )
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()

                            components.html(html_content, height=800)
                            os.remove(html_file)
                        except Exception as e:
                            st.write(f"ERROR: {e}")
                    else:
                        st.write("Please select a disease from dropdown.")
                else:
                    st.write("No diseases found in search results retrieved from SPOKE Graph.")
        # elif uploaded_file.type == "application/pdf":
        #     # Handle PDF file upload
        #     st.write("Uploaded PDF file content:")
        #     pdf_doc = fitz.open(file_path)
        #     for page_num in range(len(pdf_doc)):
        #         page = pdf_doc.load_page(page_num)
        #         text = page.get_text()
        #         st.text_area(f"Page {page_num + 1}", text, height=200)

        # elif uploaded_file.type == "text/plain":
        #     # Handle plain text file upload
        #     st.write("Uploaded Text file content:")
        #     with open(file_path, "r") as f:
        #         content = f.read()
        #         st.text_area("Text Content", content, height=200)

    else:
        st.info("Please upload a file to proceed.")


if __name__=="__main__":
    main()
