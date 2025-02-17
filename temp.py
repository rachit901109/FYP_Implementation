from spoke import SpokeNetwork, get_search
import streamlit as st
import streamlit.components.v1 as components
from pprint import pprint
from utils import save_file
import os

def main():
    st.set_page_config(layout="wide")
    st.title("Multi-Tab Graph Application")

    uploaded_file = st.file_uploader("Upload an Image, PDF, or Text file", type=["jpg", "jpeg", "png", "pdf", "txt"])

    if uploaded_file is not None:
        file_path = save_file(uploaded_file)
        st.success(f"File successfully uploaded and saved to: {file_path}")

        # Handle different file types
        if uploaded_file.type.startswith("image"):
            # Handle image file upload
            st.image(file_path, caption="Uploaded Image", use_container_width=True)

            if st.button("fuck this button"):
                dis_input = 'Melanocytic Nevi'
                network = SpokeNetwork()
                search_results = get_search(dis_input)
                disease_entries = {
                    item['name']:item for item in search_results if item['node_type'] == "Disease"
                }
                neighborhood = { dis['identifier']:network.fetch_neighborhood(
                    "Disease", 
                    "identifier", 
                    dis['identifier']
                ) for name, dis in disease_entries.items()
                }

                # Create tabs
                tab_a, tab_b, tab_c = st.tabs(["Tab A", "Tab B", "Tab C"])

                # Content for Tab A
                with tab_a:
                    st.header("Welcome to Tab A")
                    st.write("This is the content for Tab A")

                # Content for Tab B
                with tab_b:
                    st.header("Welcome to Tab B")
                    st.write("This is the content for Tab B")

                # Content for Tab C with graph selection
                with tab_c:
                    st.header("Graph Selection")

                    options = [dis.get('name', "no name key") for name, dis in disease_entries.items()]
                    options.insert(0, "")
                    selected_graph = st.selectbox(
                        "Choose a graph to display:",
                        options=options
                    )

                    dis = disease_entries.get(selected_graph, "")
                    # Display selected graph only if one is selected
                    if selected_graph!="":
                        neighborhood_data = neighborhood.get(dis['identifier'], [])
                        # Create and display network visualization
                        html_file = network.create_network_visualization(
                            neighborhood_data,
                            dis['name']
                        )
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        components.html(html_content, height=800)
                        os.remove(html_file)

if __name__ == "__main__":
    main()

