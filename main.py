import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="Cairo Transport Optimization", layout="wide")

# --- Sidebar Navigation ---
pages = [
    "Home",
    "Road Network View",
    "Traffic Flow Optimization",
    "Emergency Routing",
    "Public Transit Optimization",
    "Data Management",
    "Performance Analysis",
    "About"
]
selected_page = st.sidebar.radio("Go to", pages)

# --- Shared Utilities ---
def load_area_names():
    df = pd.read_csv("area_names.csv")
    return dict(zip(df["ID"], df["Name"]))

def load_area_locations():
    df = pd.read_csv("area_names.csv")
    return dict(zip(df["ID"], zip(df["Latitude"], df["Longitude"])))

# --- Page: Home ---
if selected_page == "Home":
    st.title("🚀 Cairo Transport Optimization System")
    st.markdown("""
    Welcome to the **Cairo Transport Optimization System**, a smart planning and analysis tool designed to improve the public and private transportation networks across Greater Cairo.

    ### 🔍 What This Platform Offers:
    - **🛣️ Road Network Visualization**: View existing and planned road layouts.
    - **🚦 Traffic Flow Optimization**: Identify efficient driving routes based on real-time traffic data.
    - **🚑 Emergency Routing**: Compute the fastest response paths for emergency services.
    - **🚍 Public Transit Optimization**: Suggest optimal bus and metro routes between areas.
    - **📊 Performance Analysis**: Compare route quality and efficiency under different scenarios.
    - **📂 Data Management**: Upload, view, and manage key datasets.

    ### 🧠 How It Works:
    The platform uses **graph theory algorithms**, **network analysis**, and **interactive maps** to:
    - Model the transportation infrastructure of Cairo.
    - Simulate different usage scenarios.
    - Suggest route and policy improvements.

    ### 🏙️ Impact:
    - Reduce congestion and travel times.
    - Improve public transit accessibility.
    - Support emergency services with dynamic routing.
    - Assist urban planners and decision-makers.

    ---
    **Get started by choosing a page from the sidebar!**
    """)

    st.image("https://cdn.britannica.com/49/94449-050-38762C60/Cairo.jpg", caption="Aerial view of Cairo", use_column_width=True)


# --- Page: Road Network View ---
elif selected_page == "Road Network View":
    st.title("🛣️ Road Network Visualization")
    st.markdown("Visualize the current and proposed road networks with traffic data overlays.")

    try:
        with open("cairo_traffic_map.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600)
    except FileNotFoundError:
        st.warning("Traffic map not found. Please generate it first using your weighted graph code.")

# --- Page: Traffic Flow Optimization ---
elif selected_page == "Traffic Flow Optimization":
    st.title("🚦 Traffic Flow Optimization")
    st.markdown("Analyze shortest and most efficient routes based on current traffic conditions.")

    st.selectbox("Select Origin Area", options=["(data needed)"])
    st.selectbox("Select Destination Area", options=["(data needed)"])
    st.selectbox("Select Time of Day", options=["Morning", "Afternoon", "Evening", "Night"])
    st.button("Run Optimization")

    st.empty()  # Placeholder for results

# --- Page: Emergency Routing ---
elif selected_page == "Emergency Routing":
    st.title("🚑 Emergency Routing")
    st.markdown("Compute the fastest emergency routes to hospitals or services based on current traffic.")

    st.selectbox("Select Incident Location", options=["(data needed)"])
    st.selectbox("Select Emergency Type", options=["Ambulance", "Fire", "Police"])
    st.button("Calculate Route")

    st.empty()  # Placeholder for route info

# --- Page: Public Transit Optimization ---
elif selected_page == "Public Transit Optimization":
    st.title("🚍 Public Transit Optimization")
    st.markdown("Optimize metro and bus networks using algorithmic scheduling.")

    # Load CSV data
    try:
        buses_df = pd.read_csv("bus_routes.csv")
        metro_df = pd.read_csv("metro_lines.csv")
        roads_df = pd.read_csv("existing_roads.csv")
        area_df = pd.read_csv("area_names.csv")

        area_dict = dict(zip(area_df["ID"], area_df["Name"]))
        coords = dict(zip(area_df["ID"], zip(area_df["Latitude"], area_df["Longitude"])))

        # Build the graph
        G = nx.Graph()
        for _, row in roads_df.iterrows():
            G.add_edge(str(row["FromID"]).strip(), str(row["ToID"]).strip(), weight=row["Distance(km)"], mode="road")

        for _, row in buses_df.iterrows():
            stops = [s.strip() for s in row["Stops(comma-separated IDs)"].strip('"').split(",")]
            for i in range(len(stops) - 1):
                G.add_edge(stops[i], stops[i + 1], weight=1, mode="bus", route=row["RouteID"])

        for _, row in metro_df.iterrows():
            stations = [s.strip() for s in row["Stations(comma-separated IDs)"].strip('"').split("/")]
            for i in range(len(stations) - 1):
                G.add_edge(stations[i], stations[i + 1], weight=0.5, mode="metro", route=row["LineID"])

        # Dropdowns for source and destination
        area_options = list(area_dict.items())
        source = st.selectbox("Select Starting Point", area_options, format_func=lambda x: x[1])
        dest = st.selectbox("Select Destination Point", area_options, format_func=lambda x: x[1])

        if st.button("Optimize Transit Route"):
            source_id = str(source[0])
            dest_id = str(dest[0])
            try:
                path = nx.shortest_path(G, source=source_id, target=dest_id, weight="weight")
                st.session_state["transit_path"] = path
            except nx.NetworkXNoPath:
                st.error("No transit path found between the selected areas.")
                st.session_state["transit_path"] = None

        # Show path if available
        if "transit_path" in st.session_state and st.session_state["transit_path"]:
            st.subheader("Optimized Transit Route")
            steps = []
            path = st.session_state["transit_path"]
            for i in range(len(path) - 1):
                edge = G[path[i]][path[i + 1]]
                mode = edge["mode"]
                route = edge.get("route", "N/A")
                steps.append(f"{area_dict[path[i]]} → {area_dict[path[i + 1]]} via {mode.upper()} ({route})")
            st.markdown("### Step-by-Step Instructions:")
            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}")
    except FileNotFoundError:
        st.error("Required transit data not found. Please ensure all CSV files are available.")


# --- Page: Data Management ---
elif selected_page == "Data Management":
    st.title("📂 Data Management")
    st.markdown("Upload or preview the datasets used for analysis.")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

# --- Page: Performance Analysis ---
elif selected_page == "Performance Analysis":
    st.title("📊 Performance Analysis")
    st.markdown("Compare algorithmic performance across various scenarios.")

    st.empty()  # Placeholder for performance metrics

# --- Page: About ---
elif selected_page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    **Project:** Cairo Transport Optimization

    **Team:** CSE112

    **Technologies Used:** Python, Streamlit, NetworkX, Folium

    **Purpose:** Enhance urban mobility and infrastructure planning in Cairo.
    """)