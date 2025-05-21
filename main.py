import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from collections import defaultdict
from cairo_road_network import CairoRoadNetwork
import heapq

# --- Page Config ---
st.set_page_config(page_title="Cairo Transport Optimization", layout="wide")

# --- Sidebar Navigation ---
pages = [
    "Home",
    "Road Network View",
    "Infrastructure Road Network",
    "Traffic Flow Optimization",
    "Emergency Routing",
    "Public Transit Optimization",
    "About"
]
selected_page = st.sidebar.radio("Go to", pages)

# --- Shared Utilities ---
def load_area_names():
    df = pd.read_csv("area_names.csv", dtype={"ID": str})
    df["ID"] = df["ID"].str.strip()
    return dict(zip(df["ID"], df["Name"]))

area_names_dict = load_area_names()


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
    - **🛣️ Infrastructure Road Network**: Identify efficient driving routes based on real-time traffic data.
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

    st.image("img.jpg", caption="Aerial view of Cairo", use_column_width=True)


# --- Page: Road Network View ---
elif selected_page == "Road Network View":
    st.title("🛣️ Road Network Visualization")
    st.markdown("Visualize the current and proposed road networks with traffic data overlays.")

    try:
        with open("cairo_traffic_map.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600)
    except FileNotFoundError:
        st.warning("Traffic map not found. Please generate it first using your weighted graph code.")

            ## ازاي حط هنا النودز بتاعتنا فين الكود الي بيدل علي دا ؟
            
# --- Page: Infrastructure Road Network ---
elif selected_page == "Infrastructure Road Network":
    st.title("🛣️ Infrastructure Road Network Visualization")
    st.markdown("Visualize the current and proposed road networks with traffic data overlays.")

    # Initialize and load network data
    network = CairoRoadNetwork()
    network.load_data()
    mst_edges = network.kruskal_mst_with_critical_backbone()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Interactive Map", "Network Analysis"])
    
    with tab1:
        st.subheader("Interactive Network Map")
        st.markdown("""
        **Map Key:**
        - 🟨 **Yellow lines:** MST (Minimum Spanning Tree) network edges
        - 🟧 **Orange lines:** Other network connections
        - ⭐ **Red star pin:** Critical facility
        - 📍 **Red location pin:** Population center
        """)
        m, _ = network.create_network_map(mst_edges)
        # Center the map in the page with wider width
        col1, col2, col3 = st.columns([0.5,3,0.5])
        with col2:
            st_folium(m, width=1200, height=700, returned_objects=None)
    
    with tab2:
        st.subheader("Network Analysis")
        
        # Display basic analysis
        st.markdown("### Basic Network Statistics")
        basic_analysis = network.analyze_network(mst_edges)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Areas", basic_analysis['total_nodes'])
            st.metric("Total Road Connections", basic_analysis['total_edges'])
        with col2:
            st.metric("Essential Road Connections (MST)", basic_analysis['mst_edges'])
            st.metric("Total Population Served", f"{basic_analysis['total_population']:,}")
        
        # Display enhanced analysis
        st.markdown("### Road Network Cost Analysis")
        enhanced_analysis = network.enhanced_cost_analysis(mst_edges)
        df_cost = pd.DataFrame(enhanced_analysis['cost_breakdown'])
        # Map edge IDs to names and rename columns
        def edge_to_names(edge):
            u, v = edge.split('-')
            return f"{area_names_dict.get(u, u)} - {area_names_dict.get(v, v)}"
        if 'edge' in df_cost:
            df_cost['edge'] = df_cost['edge'].apply(edge_to_names)
            df_cost = df_cost.rename(columns={
                'edge': 'Road Connection',
                'distance': 'Distance (km)',
                'population': 'Population Served',
                'is_critical': 'Connects Critical Facility',
                'cost_per_km': 'Cost per Kilometer'
            })
        st.dataframe(df_cost)
        
        # Display connectivity analysis
        st.markdown("### Area Connectivity Analysis")
        connectivity = network.analyze_neighborhood_connectivity(mst_edges)
        df_conn = pd.DataFrame(connectivity['neighborhood_stats'])
        if 'node' in df_conn:
            df_conn['node'] = df_conn['node'].apply(lambda x: area_names_dict.get(x, x))
            df_conn = df_conn.rename(columns={
                'node': 'Area Name',
                'mst_connections': 'Essential Connections',
                'total_connections': 'Total Connections',
                'connectivity_ratio': 'Connection Efficiency'
            })
        st.dataframe(df_conn)

# --- Page: Traffic Flow Optimization ---
elif selected_page == "Traffic Flow Optimization":
    st.title("🚦 Traffic Flow Optimization")
    st.markdown("Analyze shortest and most efficient routes based on current traffic conditions.")

    class CairoTransportationOptimizer:
        def __init__(self):
            self.areas = {}  # area_id -> name
            self.traffic_data = {}  # road_id -> {time_period: volume}
            self.time_periods = ['Morning Peak', 'Afternoon', 'Evening Peak', 'Night']
            self.graph = None

        def load_data(self, areas_file, traffic_file):
            areas_df = pd.read_csv(areas_file)
            traffic_df = pd.read_csv(traffic_file)

            self.areas = {str(row['ID']): row['Name'] for _, row in areas_df.iterrows()}
            self.traffic_data = {}

            for _, row in traffic_df.iterrows():
                road_id = row['RoadID']
                self.traffic_data[road_id] = {
                    'Morning Peak': row['Morning Peak(veh/h)'],
                    'Afternoon': row['Afternoon(veh/h)'],
                    'Evening Peak': row['Evening Peak(veh/h)'],
                    'Night': row['Night(veh/h)']
                }

        def build_graph(self, time_period):
            self.graph = nx.Graph()
            for road_id, traffic in self.traffic_data.items():
                try:
                    source, dest = road_id.split('-')
                    weight = traffic[time_period]
                    self.graph.add_edge(source, dest, weight=weight)
                except:
                    continue

        def dijkstra(self, start, end):
            if self.graph is None:
                return [], float('inf')

            try:
                path = nx.dijkstra_path(self.graph, source=start, target=end, weight='weight')
                distance = nx.dijkstra_path_length(self.graph, source=start, target=end, weight='weight')
                return path, distance
            except:
                return [], float('inf')

        def find_alternate_routes(self, start, end, closed_road=None):
            results = []
            temp_graph = self.graph.copy()

            if closed_road:
                try:
                    a, b = closed_road.split('-')
                    temp_graph.remove_edge(a, b)
                except:
                    pass

            try:
                all_paths = list(nx.all_simple_paths(temp_graph, start, end, cutoff=6))
            except:
                return results

            for path in all_paths[:5]:
                dist = sum(
                    temp_graph[path[i]][path[i+1]]['weight']
                    for i in range(len(path)-1)
                )
                results.append({"path": path, "distance": dist})

            results.sort(key=lambda x: x['distance'])
            return results

        def analyze_congestion(self, time_period):
            congestion_map = defaultdict(int)

            for road_id, traffic in self.traffic_data.items():
                source, dest = road_id.split('-')
                volume = traffic[time_period]
                congestion_map[source] += volume
                congestion_map[dest] += volume

            sorted_congestion = sorted(congestion_map.items(), key=lambda x: x[1], reverse=True)
            return sorted_congestion

        def optimize_traffic_signals(self, time_period):
            congestion = self.analyze_congestion(time_period)[:5]
            recommendations = []

            for area_id, volume in congestion:
                neighbors = []
                for road_id, traffic in self.traffic_data.items():
                    if area_id in road_id:
                        parts = road_id.split('-')
                        other = parts[1] if parts[0] == area_id else parts[0]
                        neighbors.append((other, traffic[time_period]))

                total = sum([v for _, v in neighbors])
                if total == 0:
                    continue

                signal_timing = [(n, round(60 * (v / total))) for n, v in neighbors]
                recommendations.append({
                    "name": self.areas.get(area_id, area_id),
                    "congestion": volume,
                    "signal_timing": signal_timing
                })

            return recommendations

        def suggest_congestion_reduction(self, time_period):
            congestion = self.analyze_congestion(time_period)[:5]
            suggestions = []

            for area_id, volume in congestion:
                strategies = [
                    "Divert traffic to alternative routes",
                    "Implement temporary one-way streets",
                    "Increase public transport availability",
                    "Use dynamic lane management"
                ]
                suggestions.append({
                    "area": self.areas.get(area_id, area_id),
                    "congestion": volume,
                    "strategies": strategies
                })

            return suggestions

        def visualize_network(self, path=None, closed_road=None, time_period='Morning Peak'):
            pos = nx.spring_layout(self.graph, seed=42)
            fig, ax = plt.subplots(figsize=(10, 7))

            edge_colors = []
            for u, v in self.graph.edges():
                road = f"{u}-{v}" if f"{u}-{v}" in self.traffic_data else f"{v}-{u}"
                traffic = self.traffic_data[road][time_period] if road in self.traffic_data else 0
                edge_colors.append(traffic)

            nx.draw(
                self.graph, pos, ax=ax, with_labels=True,
                labels={n: self.areas.get(n, n) for n in self.graph.nodes()},
                edge_color=edge_colors, width=2, edge_cmap=plt.cm.coolwarm,
                node_size=700, node_color='lightblue', font_size=9
            )

            if path:
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='black', width=3, ax=ax)

            if closed_road:
                try:
                    u, v = closed_road.split('-')
                    nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], edge_color='red', width=4, ax=ax)
                except:
                    pass

            plt.title(f"Traffic Network Visualization - {time_period}")
            return fig

    try:
        # Initialize the optimizer
        optimizer = CairoTransportationOptimizer()
        optimizer.load_data("area_names.csv", "traffic_flow.csv")

        # UI Components
        st.markdown("### 🚗 Route Planning")
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period",
            ['Morning Peak', 'Afternoon', 'Evening Peak', 'Night']
        )
        
        # Build graph for selected time period
        optimizer.build_graph(time_period)

        # Source and destination selection
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Select Starting Point", list(optimizer.areas.items()), format_func=lambda x: x[1])
        with col2:
            destination = st.selectbox("Select Destination", list(optimizer.areas.items()), format_func=lambda x: x[1])

        # Closed roads selection
        closed_roads = st.multiselect(
            "Select Closed Roads (optional)",
            list(optimizer.traffic_data.keys())
        )

        if st.button("Calculate Route"):
            # Find primary route
            path, distance = optimizer.dijkstra(source[0], destination[0])
            
            if path:
                st.success("✅ Route found!")
                
                # Display the visualization
                fig = optimizer.visualize_network(path=path, closed_road=closed_roads[0] if closed_roads else None, time_period=time_period)
                st.pyplot(fig)
                
                # Show route details
                st.markdown("### 📍 Route Details")
                st.markdown(f"**Distance:** {distance:.2f} units")
                st.markdown("**Path:**")
                path_names = [optimizer.areas.get(node, node) for node in path]
                st.markdown(" → ".join(path_names))
                
                # Find alternative routes
                st.markdown("### 🔄 Alternative Routes")
                alt_routes = optimizer.find_alternate_routes(source[0], destination[0], closed_roads[0] if closed_roads else None)
                
                for i, route in enumerate(alt_routes[1:], 1):  # Skip the first one as it's similar to primary
                    st.markdown(f"**Route {i}**")
                    route_names = [optimizer.areas.get(node, node) for node in route["path"]]
                    st.markdown(f"Distance: {route['distance']:.2f} units")
                    st.markdown(" → ".join(route_names))
            else:
                st.error("❌ No valid route found between selected points")

        # Traffic Analysis Section
        st.markdown("### 🚥 Traffic Analysis")
        
        if st.button("Analyze Traffic"):
            # Show congestion analysis
            congestion = optimizer.analyze_congestion(time_period)
            st.markdown("#### Most Congested Areas")
            for area_id, volume in congestion[:5]:
                st.markdown(f"- **{optimizer.areas.get(area_id, area_id)}**: {volume:.0f} vehicles/hour")

            # Show traffic signal recommendations
            st.markdown("#### 🚦 Traffic Signal Recommendations")
            recommendations = optimizer.optimize_traffic_signals(time_period)
            for rec in recommendations:
                st.markdown(f"**{rec['name']}** (Congestion: {rec['congestion']:.0f} vehicles/hour)")
                for node, timing in rec['signal_timing']:
                    st.markdown(f"- {optimizer.areas.get(node, node)}: {timing} seconds")

            # Show congestion reduction suggestions
            st.markdown("#### 💡 Congestion Reduction Strategies")
            suggestions = optimizer.suggest_congestion_reduction(time_period)
            for sugg in suggestions:
                st.markdown(f"**{sugg['area']}** (Congestion: {sugg['congestion']:.0f} vehicles/hour)")
                for strategy in sugg['strategies']:
                    st.markdown(f"- {strategy}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# --- Page: Emergency Routing ---
elif selected_page == "Emergency Routing":
    st.title("🚑 Emergency Routing Using A* Algorithm")

    # --- Load data ---
    @st.cache_data
    def load_data():
        area_df = pd.read_csv("area_names.csv")
        roads_df = pd.read_csv("existing_roads.csv")
        traffic_df = pd.read_csv("traffic_flow.csv")
        facilities_df = pd.read_csv("facilities.csv")

        area_dict = dict(zip(area_df["ID"].astype(str), area_df["Name"].str.strip()))
        coords = dict(zip(area_df["ID"].astype(str), zip(area_df["Latitude"], area_df["Longitude"])))

        hospitals = facilities_df[facilities_df["Type"].str.strip() == "Medical"]
        hospital_coords = dict(zip(hospitals["ID"].astype(str), zip(hospitals["Y-coordinate"], hospitals["X-coordinate"])))
        hospital_names = dict(zip(hospitals["ID"].astype(str), hospitals["Name"].str.strip()))

        traffic_map = {
            row["RoadID"].strip(): {
                "Morning": float(row["Morning Peak(veh/h)"]),
                "Afternoon": float(row["Afternoon(veh/h)"]),
                "Evening": float(row["Evening Peak(veh/h)"]),
                "Night": float(row["Night(veh/h)"]),
            }
            for _, row in traffic_df.iterrows()
        }

        return roads_df, area_dict, coords, hospital_coords, hospital_names, traffic_map, hospitals

    roads_df, area_dict, coords, hospital_coords, hospital_names, traffic_map, hospitals = load_data()

    # --- UI Inputs ---
    area_options = list(area_dict.items())
    source = st.selectbox("🚨 Select Incident Location", area_options, format_func=lambda x: x[1])

    hospital_options = [None] + list(hospital_names.items())
    destination = st.selectbox("🏥 Select Destination Hospital (optional)", hospital_options,
                               format_func=lambda x: "--- Auto Select Closest ---" if x is None else x[1])

    time_option = st.selectbox("🕒 Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    # --- Build graph with weights considering traffic ---
    graph = {}
    for _, row in roads_df.iterrows():
        from_id = str(row["FromID"]).strip()
        to_id = str(row["ToID"]).strip()
        try:
            distance = float(row["Distance(km)"])
        except:
            continue
        if pd.isna(distance) or distance <= 0:
            continue

        road_id_1 = f"{from_id}-{to_id}"
        road_id_2 = f"{to_id}-{from_id}"

        traffic = traffic_map.get(road_id_1) or traffic_map.get(road_id_2)
        weight = distance * (1 + (traffic[time_option] / 3000)) if traffic else distance * 2

        if from_id not in graph:
            graph[from_id] = []
        if to_id not in graph:
            graph[to_id] = []

        graph[from_id].append((to_id, weight))
        graph[to_id].append((from_id, weight))

    # --- Heuristic function: Euclidean distance ---
    def heuristic(u, v):
        u_coords = coords.get(u) or hospital_coords.get(u, (0, 0))
        v_coords = hospital_coords.get(v, (0, 0))
        return ((u_coords[0] - v_coords[0]) ** 2 + (u_coords[1] - v_coords[1]) ** 2) ** 0.5

    # --- A* algorithm implementation ---
    def astar_search(graph, start, goal, heuristic):
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
        visited = set()
        cost_so_far = {start: 0}

        while open_set:
            f, g, current, path = heapq.heappop(open_set)

            if current == goal:
                return path, g

            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in graph.get(current, []):
                new_cost = g + weight
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))

        return None, float('inf')

    # --- Run A* search ---
    if st.button("Calculate Emergency Route"):
        source_id = str(source[0])

        if destination is not None:
            hospital_id = str(destination[0])
            path, cost = astar_search(graph, source_id, hospital_id, heuristic)
            if path is None:
                st.error("❌ No path found to the selected hospital.")
                st.session_state.emergency_result = None
            else:
                st.session_state.emergency_result = (hospital_id, path, cost)
        else:
            best_path = None
            best_cost = float('inf')
            best_hospital = None
            for hospital_id in hospitals["ID"].astype(str):
                path, cost = astar_search(graph, source_id, hospital_id, heuristic)
                if path is not None and cost < best_cost:
                    best_path = path
                    best_cost = cost
                    best_hospital = hospital_id
            if best_path:
                st.session_state.emergency_result = (best_hospital, best_path, best_cost)
            else:
                st.error("❌ No path found to any hospital.")
                st.session_state.emergency_result = None

    # --- Show results ---
    if "emergency_result" in st.session_state and st.session_state.emergency_result:
        hospital_id, best_path, best_cost = st.session_state.emergency_result
        st.success(f"✅ Emergency route to: *{hospital_names[hospital_id]}*")

        st.markdown("### Step-by-Step Directions:")

        def choose_green_light(data):
            sorted_directions = sorted(
                data.items(),
                key=lambda item: (not item[1]["has_ambulance"], -item[1]["vehicles"])
            )
            return sorted_directions[0][0]

        def get_direction(from_coord, to_coord):
            lat1, lon1 = from_coord
            lat2, lon2 = to_coord
            if abs(lat2 - lat1) > abs(lon2 - lon1):
                return "North" if lat2 > lat1 else "South"
            else:
                return "East" if lon2 > lon1 else "West"

        for i in range(len(best_path) - 1):
            from_node = best_path[i]
            to_node = best_path[i + 1]
            from_name = area_dict.get(from_node, hospital_names.get(from_node, from_node))
            to_name = area_dict.get(to_node, hospital_names.get(to_node, to_node))
            st.markdown(f"{i + 1}.** {from_name} → {to_name}")

            from_coords = coords.get(from_node) or hospital_coords.get(from_node)
            to_coords = coords.get(to_node) or hospital_coords.get(to_node)
            actual_direction = get_direction(from_coords, to_coords)

            intersection_sim = {
                "North": {"vehicles": 12, "has_ambulance": False},
                "South": {"vehicles": 15, "has_ambulance": False},
                "East": {"vehicles": 8, "has_ambulance": True},
                "West": {"vehicles": 10, "has_ambulance": False}
            }

            chosen_direction = choose_green_light(intersection_sim)

            if actual_direction == chosen_direction:
                signal_note = "🚦 ✅ Signal already green for this direction"
            else:
                signal_note = "🚦 ⚠ Signal changed to green for ambulance (preemption)"

            st.info(signal_note)

        st.markdown(f"### 📊 Estimated Emergency Travel Cost: {best_cost:.2f} units")

        # --- Draw route on map (one clean polyline only) ---
        route_coords = [coords.get(n) or hospital_coords.get(n) for n in best_path]
        route_map = folium.Map(location=route_coords[0], zoom_start=12)
        folium.PolyLine(route_coords, color="red", weight=6).add_to(route_map)

        for i, node in enumerate(best_path):
            loc = coords.get(node) or hospital_coords.get(node)
            name = area_dict.get(node) or hospital_names.get(node)
            icon_color = "red" if node == hospital_id else "blue"
            folium.Marker(loc, popup=name, icon=folium.Icon(color=icon_color)).add_to(route_map)

            if i < len(best_path) - 1:
                next_node = best_path[i + 1]
                from_coords = coords.get(node) or hospital_coords.get(node)
                to_coords = coords.get(next_node) or hospital_coords.get(next_node)
                actual_direction = get_direction(from_coords, to_coords)
                chosen_direction = choose_green_light(intersection_sim)

                if actual_direction == chosen_direction:
                    signal_popup = "🚦 ✅ Signal already green for this direction"
                else:
                    signal_popup = "🚦 ⚠ Signal changed to green for ambulance (preemption)"

            folium.CircleMarker(
                location=loc,
                radius=8,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=1,
                popup=signal_popup
            ).add_to(route_map)
        st_folium(route_map, height=500)


# --- Page: Public Transit Optimization ---
elif selected_page == "Public Transit Optimization":
    st.title("🚍 Public Transit Optimization")
    
    # Initialize session state variables if they don't exist
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'has_calculated' not in st.session_state:
        st.session_state.has_calculated = False

    st.markdown("""
    ### Dynamic Programming Approach to Transit Optimization
    This system uses dynamic programming to:
    - Optimize routes considering multiple transport modes
    - Minimize total travel time including transfers
    - Balance between speed and number of transfers
    """)

    try:
        buses_df = pd.read_csv("bus_routes.csv")
        metro_df = pd.read_csv("metro_lines.csv")
        roads_df = pd.read_csv("existing_roads.csv")
        area_df = pd.read_csv("area_names.csv")

        # Normalize node IDs to strings and strip any whitespace
        area_df["ID"] = area_df["ID"].astype(str).str.strip()
        area_dict = dict(zip(area_df["ID"], area_df["Name"]))
        coords_dict = dict(zip(area_df["ID"], zip(area_df["Latitude"], area_df["Longitude"])))
        valid_ids = set(area_dict.keys())

        def minutes_to_time_str(minutes):
            hours = int(minutes) // 60
            mins = int(minutes) % 60
            return f"{hours:02d}:{mins:02d}"

        def time_str_to_minutes(time_str):
            try:
                hours, minutes = map(int, time_str.split(":"))
                return hours * 60 + minutes
            except:
                return 480  # Default to 08:00 if invalid

        # Create adjacency list representation
        adj_list = {}
        for id in valid_ids:
            adj_list[id] = []

        # Add road connections
        for _, row in roads_df.iterrows():  #make connections between roads
            u = str(row["FromID"]).strip()
            v = str(row["ToID"]).strip()
            if u in valid_ids and v in valid_ids:
                dist = float(row["Distance(km)"])
                adj_list[u].append((v, dist, "road", "direct", 0))
                adj_list[v].append((u, dist, "road", "direct", 0))

        # Add bus connections
        for _, row in buses_df.iterrows():  #make connections between buses
            stops = [s.strip() for s in str(row["Stops(comma-separated IDs)"]).strip('"').split(",")]
            freq = int(row.get("Frequency(mins)", 15))
            route_id = row["RouteID"]
            for i in range(len(stops) - 1):
                u, v = stops[i], stops[i + 1]
                if u in valid_ids and v in valid_ids:
                    adj_list[u].append((v, 1, "bus", route_id, freq)) #################
                    adj_list[v].append((u, 1, "bus", route_id, freq))

        # Add metro connections
        for _, row in metro_df.iterrows():   #make connections between metro lines
            stations = [s.strip() for s in str(row["Stations(comma-separated IDs)"]).strip('"').split("/")]
            freq = int(row.get("Frequency(mins)", 5))
            line_id = row["LineID"]
            for i in range(len(stations) - 1):
                u, v = stations[i], stations[i + 1]
                if u in valid_ids and v in valid_ids:
                    adj_list[u].append((v, 0.5, "metro", line_id, freq)) ################
                    adj_list[v].append((u, 0.5, "metro", line_id, freq))

        # UI for selecting source/destination
        area_options = list(area_dict.items())
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Select Starting Point", area_options, format_func=lambda x: x[1], key='source')
        with col2:
            dest = st.selectbox("Select Destination Point", area_options, format_func=lambda x: x[1], key='dest')
        
        # Time input in HH:MM format
        default_time = "08:00"
        time_str = st.text_input("Start Time (HH:MM, 24-hour format)", value=default_time, key='time_input')
        start_time = time_str_to_minutes(time_str)

        def dynamic_transit_optimization(adj_list, source, target, start_time):
            INF = float('inf')
            n = len(adj_list)
            
            # State: (node, prev_mode, prev_route)
            # Value: (min_time, prev_state, total_transfers)
            dp = {}
            
            def get_state_key(node, prev_mode, prev_route):
                return (node, prev_mode, prev_route)
            
            def get_wait_time(curr_time, frequency):   #8:23 (83) , 15
                return frequency - (curr_time % frequency) if frequency > 0 else 0  # 15-(83 % 15)=7
            
            # Initialize DP table
            initial_state = get_state_key(source, None, None)
            dp[initial_state] = (start_time, None, 0)
            
            # Priority queue for Dijkstra-like processing
            # Format: (current_time, node, prev_mode, prev_route)
            from queue import PriorityQueue    # queue for the best states
            pq = PriorityQueue()
            pq.put((start_time, source, None, None))
            
            while not pq.empty():
                curr_time, curr_node, prev_mode, prev_route = pq.get()
                curr_state = get_state_key(curr_node, prev_mode, prev_route)
                
                # Skip if we've found a better time for this state
                if dp[curr_state][0] < curr_time:
                    continue
                
                # Process all neighbors
                for next_node, distance, mode, route, frequency in adj_list[curr_node]: #############
                    # Calculate travel time
                    travel_time = distance * 10  # Base travel time
                    
                    # Calculate waiting time
                    wait_time = 0
                    if mode != "road":  # Only wait for public transport
                        wait_time = get_wait_time(curr_time, frequency)
                    
                    # Calculate transfer penalty
                    transfer_penalty = 5 if prev_mode and mode != prev_mode else 0
                    
                    total_step_time = wait_time + travel_time + transfer_penalty
                    next_time = curr_time + total_step_time
                    
                    # Calculate transfers
                    transfers = dp[curr_state][2]
                    if prev_mode and mode != prev_mode:
                        transfers += 1
                    
                    next_state = get_state_key(next_node, mode, route)
                    if next_state not in dp or next_time < dp[next_state][0]:
                        dp[next_state] = (next_time, curr_state, transfers)
                        pq.put((next_time, next_node, mode, route))
            
            # Find best path to target
            best_final_state = None
            best_time = INF
            
            for state in dp:
                if state[0] == target:
                    time = dp[state][0]
                    if time < best_time:
                        best_time = time
                        best_final_state = state
            
            if best_final_state is None:
                return None
            
            # Reconstruct path
            path = []
            current_state = best_final_state
            while current_state is not None:
                path.append(current_state)
                current_state = dp[current_state][1]
            
            path.reverse()
            return path, dp

        def calculate_route():
            source_id = str(source[0]).strip()
            dest_id = str(dest[0]).strip()
            with st.spinner('🔄 Calculating optimal route...'):
                result = dynamic_transit_optimization(adj_list, source_id, dest_id, start_time)
                st.session_state.optimization_result = result
                st.session_state.has_calculated = True

        # Create two columns for buttons
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Optimize Route", key='optimize_button'):
                calculate_route()
        with col2:
            if st.button("Clear Results", key='clear_button'):
                st.session_state.optimization_result = None
                st.session_state.has_calculated = False
                st.experimental_rerun()

        # Show results if available
        if st.session_state.has_calculated:
            if st.session_state.optimization_result:
                path, dp = st.session_state.optimization_result
                
                with st.container():
                    st.subheader("🧭 Optimized Transit Path (Dynamic Programming)")
                    
                    # Calculate total time and transfers
                    total_time = dp[path[-1]][0] - start_time
                    total_transfers = dp[path[-1]][2]
                    
                    # Create three columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Start Time", minutes_to_time_str(start_time))
                    with col2:
                        st.metric("End Time", minutes_to_time_str(dp[path[-1]][0]))
                    with col3:
                        st.metric("Total Duration", f"{int(total_time)} min")

                    st.metric("Number of Transfers", total_transfers)
                    
                    # Create a map for visualization
                    m = folium.Map(location=[30.0444, 31.2357], zoom_start=11)
                    
                    # Draw the path on the map with different colors for different modes
                    mode_colors = {
                        "road": "gray",
                        "bus": "blue",
                        "metro": "red"
                    }
                    
                    current_time = start_time
                    
                    with st.expander("🚶 Step-by-Step Timeline", expanded=True):
                        for i in range(len(path) - 1):
                            curr_node, curr_mode, curr_route = path[i]
                            next_node, next_mode, next_route = path[i + 1]
                            
                            # Get coordinates for current and next nodes
                            if curr_node in coords_dict and next_node in coords_dict:
                                curr_coords = coords_dict[curr_node]
                                next_coords = coords_dict[next_node]
                                
                                # Add markers for stations
                                folium.Marker(
                                    curr_coords,
                                    popup=f"{area_dict[curr_node]}<br>Time: {minutes_to_time_str(current_time)}",
                                    icon=folium.Icon(color="green" if i == 0 else "blue")
                                ).add_to(m)
                                
                                if i == len(path) - 2:  # Last node
                                    folium.Marker(
                                        next_coords,
                                        popup=f"{area_dict[next_node]}",
                                        icon=folium.Icon(color="red")
                                    ).add_to(m)
                                
                                # Draw path line with mode-specific color
                                folium.PolyLine(
                                    locations=[curr_coords, next_coords],
                                    color=mode_colors.get(next_mode, "gray"),
                                    weight=3,
                                    opacity=0.8,
                                    popup=f"{next_mode.upper()} {next_route}"
                                ).add_to(m)
                            
                            # Find the connection details
                            connection = next((x for x in adj_list[curr_node] 
                                            if x[0] == next_node and x[2] == next_mode and x[3] == next_route), None)
                            
                            if connection:
                                _, distance, mode, route, frequency = connection
                                travel_time = distance * 10
                                wait_time = 0 if mode == "road" else (frequency - (current_time % frequency))
                                transfer_penalty = 5 if i > 0 and mode != path[i-1][1] else 0
                                
                                step_time = wait_time + travel_time + transfer_penalty
                                
                                st.markdown(
                                    f"**{i+1}.** {area_dict[curr_node]} → {area_dict[next_node]}  \n"
                                    f"Mode: {mode.upper()} {route}  \n"
                                    f"Start: {minutes_to_time_str(current_time)}, "
                                    f"Wait: {int(wait_time)} min, "
                                    f"Travel: {int(travel_time)} min, "
                                    f"Arrive: {minutes_to_time_str(current_time + step_time)}"
                                )
                                
                                current_time += step_time

                    with st.expander("🗺️ Route Map", expanded=True):
                        st.markdown("""
                        **Legend:**
                        - 🟢 Start Point
                        - 🔵 Transfer Points
                        - 🔴 End Point
                        - Gray Line: Road
                        - Blue Line: Bus
                        - Red Line: Metro
                        """)
                        st_folium(m, height=500)
                    
                    with st.expander("📊 Route Statistics", expanded=True):
                        mode_usage = {}
                        for _, mode, _ in path[1:]:  # Skip first node
                            mode_usage[mode] = mode_usage.get(mode, 0) + 1
                        
                        st.markdown("**Transport Mode Distribution:**")
                        for mode, count in mode_usage.items():
                            st.markdown(f"- {mode.upper()}: {count} segments")
                
            else:
                st.error("No valid path found between the selected locations.")
                
    except FileNotFoundError:
        st.error("Required data files not found. Please ensure all CSVs are in place.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- Page: About ---
elif selected_page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    **Project:** Cairo Transport Optimization

    **Team:** CSE112

    **Technologies Used:** Python, Streamlit, NetworkX, Folium

    **Purpose:** Enhance urban mobility and infrastructure planning in Cairo.
    """)