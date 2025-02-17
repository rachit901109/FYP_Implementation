import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import os
import requests
from langchain_community.retrievers import WikipediaRetriever
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from pyvis.network import Network
import streamlit.components.v1 as components
from groq import Groq
import tiktoken

def get_search(disease):
    url = f"https://spoke.rbvi.ucsf.edu/api/v1/search/{disease}"
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"error: {e}, code: {response.status_code}")
    return data

def get_wikipedia_info(diagnosis):
    """Retrieve information about the diagnosis from Wikipedia"""
    try:
        retriever = WikipediaRetriever(load_all_available_meta=True)
        docs = retriever.invoke(diagnosis)
        if docs:
            return docs[0].page_content
        return "No Wikipedia information found."
    except Exception as e:
        return f"Error retrieving Wikipedia information: {str(e)}"

class SpokeNetwork:
    def __init__(self):
        self.base_url = "https://spoke.rbvi.ucsf.edu/api/v1"
        self.graph = nx.MultiDiGraph()
        
    def fetch_neighborhood(self, node_type: str, attr: str, query: str, depth: int = 1):
        url = f"{self.base_url}/neighborhood/{node_type}/{attr}/{requests.utils.quote(query)}"
        
        payload = {
            "request":"neighborhood",
            "type":f"{node_type}",
            "attr":f"{attr}",
            "query":f"{query}",
            "depth":depth,
            "sea_cutoff":10,
            "node_filters":["Compound",
                            "Location",
                            "ProteinFamily",
                            "Cytoband",
                            "MiRNA",
                            "PwGroup",
                            "Blend",
                            "DietarySupplement",
                            "MolecularFunction",
                            "Reaction",
                            "CellLine",
                            "Disease",
                            "Organism",
                            "SDoH",
                            "CellType",
                            "EC",
                            "PanGene",
                            "SideEffect",
                            "CellularComponent",
                            "Environment",
                            "Pathway",
                            "Symptom",
                            "Chromosome",
                            "Food",
                            "PharmacologicClass",
                            "Variant",
                            "ClinicalLab",
                            "Gene",
                            "Protein",
                            "Complex",
                            "Haplotype",
                            "ProteinDomain"
                            ],
            "edge_filters":[
                "CORRELATES_AcA",
                "CONTAINS_CcG",
                "UPREGULATES_OGuG",
                "DOWNREGULATES_AdG",
                "CONTAINS_DScB",
                "CAUSES_OcD",
                "EXPRESSES_AeG",
                "CONTAINS_DScC",
                "ENCODES_OeP",
                "ISA_AiA",
                "ASSOCIATES_DaG",
                "INCLUDES_OiEC",
                "PARTOF_ApA",
                "ISA_DiD",
                "ISA_OiO",
                "UPREGULATES_AuG",
                "LOCALIZES_DlA",
                "ISOLATEDIN_OiL",
                "CONTAINS_BcC",
                "MORTALITY_DmL",
                "RESPONDS_TO_OrC",
                "CONTAINS_BcF",
                "PRESENTS_DpS",
                "ENCODES_PGeP",
                "DERIVES_FROM_CLndD",
                "PREVALENCE_DpL",
                "INCLUDES_PWiEC",
                "HAS_CLnhV",
                "RESEMBLES_DrD",
                "ISA_PWiPW",
                "ISA_CLniCLn",
                "CATALYZES_ECcR",
                "PARTICIPATES_PGpR",
                "RELATED_TO_CLnrCLn",
                "ISA_ECiEC",
                "PARTOF_PGpPG",
                "SAME_CLnsCLn",
                "FOUNDIN_EfL",
                "INCLUDES_PCiC",
                "ISA_CTiCT",
                "ISA_EiE",
                "CLEAVESTO_PctP",
                "PARTOF_CTpA",
                "CONTAINS_FcC",
                "DECREASEDIN_PdD",
                "CONTAINS_CcC",
                "ISA_FiF",
                "EXPRESSEDIN_PeCL",
                "ISA_CLiCL",
                "ASSOCIATES_GaS",
                "EXPRESSEDIN_PeCT",
                "MEASURES_CLmA",
                "ENCODES_GeM",
                "HAS_PhEC",
                "MEASURES_CLmBP",
                "ENCODES_GeP",
                "INCREASEDIN_PiD",
                "MEASURES_CLmCT",
                "EXPRESSEDIN_GeiCT",
                "INTERACTS_PiC",
                "MEASURES_CLmCC",
                "EXPRESSEDIN_GeiD",
                "INTERACTS_PiP",
                "MEASURES_CLmC",
                "MARKER_NEG_GmnD",
                "MarkerOF_PmCT",
                "MEASURES_CLmMF",
                "MARKER_POS_GmpD",
                "PARTICIPATES_PpR",
                "MEASURES_CLmO",
                "PARTICIPATES_GpBP",
                "PARTOF_PpCP",
                "MEASURES_CLmP",
                "PARTICIPATES_GpCC",
                "REGULATES_PrG",
                "AFFECTS_CamG",
                "PARTICIPATES_GpMF",
                "TRANSPORTS_PtC",
                "BINDS_CbP",
                "PARTICIPATES_GpPW",
                "INTERACTS_PDiPD",
                "BINDS_CbPD",
                "PARTICIPATES_GpR",
                "MEMBEROF_PDmPF",
                "CAUSES_CcSE",
                "PARTOF_GpPG",
                "PARTOF_PDpP",
                "CONTRAINDICATES_CcD",
                "DOWNREGULATES_GPdG",
                "CONSUMES_RcC",
                "DOWNREGULATES_CdG",
                "UPREGULATES_GPuG",
                "PARTOF_RpPW",
                "FOUNDIN_CfL",
                "BELONGS_HbG",
                "PRODUCES_RpC",
                "HASROLE_ChC",
                "DOWNREGULATES_KGdG",
                "ASSOCIATES_SaD",
                "INTERACTS_CiC",
                "UPREGULATES_KGuG",
                "ISA_SiS",
                "INTERACTS_CiF",
                "PARTOF_LpL",
                "PREVALENCEIN_SpL",
                "ISA_CiC",
                "TARGETS_MtG",
                "ASSOCIATES_VaP",
                "PARTICIPATES_CpR",
                "ADVRESPONSE_TO_mGarC",
                "BELONGS_VbH",
                "PARTOF_CpC",
                "REDUCES_SEN_mGrsC",
                "DOWNREGULATES_VdG",
                "PARTOF_CpPG",
                "RESISTANT_TO_mGrC",
                "MAPS_VmG",
                "TREATS_CtD",
                "RESPONSE_TO_mGrC",
                "UPREGULATES_VuG",
                "UPREGULATES_CuG",
                "DOWNREGULATES_OGdG"
            ],
            "cutoff_Compound_max_phase":3,
            "cutoff_Organism_Has_AMR_Phenotype_BV_BRC":"+",
            "cutoff_Organism_Host_is_Human_BV_BRC":"+",
            "cutoff_Organism_IS_Resistant":"+",
            "cutoff_Organism_USA_Isolation_Country":"+",
            "cutoff_Protein_source":"SwissProt",
            "cutoff_CtD_phase":3,
            "cutoff_DScC_quantity":700.01,
            "cutoff_DaG_diseases_sources":["knowledge","experiments"],
            "cutoff_DaG_textmining":3,
            "cutoff_DpL_average_prevalence":16.4,
            "cutoff_GeM_prediction_score":80,
            "cutoff_LINCS1000_times":["3+hours","6+hours","24+hours"],
            "cutoff_LINCS1000_tissues":"summary",
            "cutoff_PeCT_protein_markers":"yes",
            "cutoff_PeCT_reliability":"Supported",
            "cutoff_PiC_confidence":0.7,
            "cutoff_PiC_experimental_confidence":0.4,
            "cutoff_PiP_confidence":0.7
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Encoding":"gzip, deflate, br, zstd"
        }
        
        response = requests.get(url, data=payload, headers=headers)
        return response.json()

    def build_graph(self, data: List[Dict]):
        self.graph.clear()
        
        for item in data:
            if "target" not in item["data"]:
                # Node
                self.graph.add_node(
                    item["data"]["id"],
                    **item["data"].get("properties", {}),
                    node_type=item["data"].get("neo4j_type")
                )
            else:
                # Edge
                self.graph.add_edge(
                    item["data"]["source"],
                    item["data"]["target"],
                    key=item["data"]["id"],
                    edge_type=item["data"].get("neo4j_type")
                )

    def visualize(self):
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_size=1000,
                             node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        # Add labels
        labels = {node: self.graph.nodes[node].get('name', str(node)) 
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("SPOKE Network Visualization")
        plt.axis('off')
        #plt.show()
        return plt

    def create_network_visualization(self, neighborhood_data: dict, title: str) -> str:
        """Create interactive network visualization using Pyvis"""
        # Initialize the network
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        net.toggle_hide_edges_on_drag(True)
        net.show_buttons(filter_=['physics'])
        
        # Track added nodes to avoid duplicates
        added_nodes = set()
        
        # Process nodes
        for item in neighborhood_data:
            if "target" not in item["data"]:  # This is a node
                node_id = item["data"]["id"]
                if node_id not in added_nodes:
                    properties = item["data"].get("properties", {})
                    node_type = item["data"].get("neo4j_type", "Unknown")
                    
                    # Create node label
                    label = properties.get("name", str(node_id))
                    
                    # Set node color based on type
                    color = "#97C2FC"  # default blue
                    if node_type == "Disease":
                        color = "#FF9999"  # red for diseases
                    elif node_type == "Gene":
                        color = "#99FF99"  # green for genes
                    elif node_type == "Protein":
                        color = "#FFFF99"  # yellow for proteins
                    
                    # Add node with properties
                    net.add_node(
                        node_id, 
                        label=label, 
                        title=f"Type: {node_type}<br>Name: {label}", 
                        color=color
                    )
                    added_nodes.add(node_id)
        
        # Process edges
        for item in neighborhood_data:
            if "target" in item["data"]:  # This is an edge
                source = item["data"]["source"]
                target = item["data"]["target"]
                edge_type = item["data"].get("neo4j_type", "")
                
                if source in added_nodes and target in added_nodes:
                    net.add_edge(
                        source, 
                        target, 
                        title=edge_type,
#                        physics=True
                    )
        
        # Save the network to HTML file
        html_file = f"network_{title.lower().replace(' ', '_')}.html"
        net.toggle_physics(False)
        net.save_graph(html_file)
        
        return html_file
