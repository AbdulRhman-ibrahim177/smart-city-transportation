
# import csv
# from collections import defaultdict
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ======================
# # STEP 0: Read area names from CSV
# # ======================
# def read_area_names(filename):
#     area_names = {}
#     with open(filename, "r", encoding="utf-8") as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             area_id = row["ID"].strip()
#             name = row["Name"].strip()
#             area_names[area_id] = name
#     return area_names

# area_names = read_area_names("area_names.csv")

# # ======================
# # STEP 1: Read roads and traffic data
# # ======================
# def read_roads_from_csv(filename, from_col, to_col, distance_col):
#     roads = []
#     with open(filename, "r", encoding="utf-8") as file:
#         reader = csv.DictReader(file)
#         reader.fieldnames = [header.strip() for header in reader.fieldnames]
#         for row in reader:
#             from_id = row[from_col].strip()
#             to_id = row[to_col].strip()
#             distance = float(row[distance_col].strip())
#             roads.append((from_id, to_id, distance))
#     return roads

# def read_traffic_data(filename):
#     traffic_data = {}
#     with open(filename, "r", encoding="utf-8") as file:
#         reader = csv.DictReader(file)
#         reader.fieldnames = [header.strip() for header in reader.fieldnames]
#         for row in reader:
#             road_id = row["RoadID"].strip()
#             try:
#                 traffic_data[road_id] = {
#                     "Morning": float(row["Morning Peak(veh/h)"].strip()),
#                     "Afternoon": float(row["Afternoon(veh/h)"].strip()),
#                     "Evening": float(row["Evening Peak(veh/h)"].strip()),
#                     "Night": float(row["Night(veh/h)"].strip()),
#                 }
#             except ValueError:
#                 print(f"⚠️ Error reading traffic data for road {road_id}, skipping.")
#                 traffic_data[road_id] = {"Morning": None, "Afternoon": None, "Evening": None, "Night": None}
#     return traffic_data

# existing_roads = read_roads_from_csv("existing_roads.csv", "FromID", "ToID", "Distance(km)")
# new_roads = read_roads_from_csv("new_roads.csv", "FromID", "ToID", "Distance(km)")
# traffic_data = read_traffic_data("traffic_flow.csv")
# all_roads = existing_roads + new_roads

# # ======================
# # STEP 2: Ask for time of day
# # ======================
# print("⏰ Available times: Morning, Afternoon, Evening, Night")
# selected_time = input("Enter desired time for traffic visualization: ").capitalize()

# if selected_time not in ["Morning", "Afternoon", "Evening", "Night"]:
#     print("❌ Invalid time! Using 'Morning' as default.")
#     selected_time = "Morning"

# # ======================
# # STEP 3: Build Graph
# # ======================
# G = nx.Graph()

# for from_id, to_id, distance in all_roads:
#     road_id = f"{from_id}-{to_id}" if f"{from_id}-{to_id}" in traffic_data else f"{to_id}-{from_id}"
#     traffic = traffic_data.get(road_id, {"Morning": None, "Afternoon": None, "Evening": None, "Night": None})
#     G.add_edge(from_id, to_id, weight=distance, traffic=traffic)

# # ======================
# # STEP 4: Define traffic ranges
# # ======================
# traffic_ranges = {
#     "Morning": {"low": (0, 2000), "medium": (2001, 3000), "high": (3001, float('inf'))},
#     "Afternoon": {"low": (0, 1200), "medium": (1201, 1800), "high": (1801, float('inf'))},
#     "Evening": {"low": (0, 1500), "medium": (1501, 2400), "high": (2401, float('inf'))},
#     "Night": {"low": (0, 300), "medium": (301, 600), "high": (601, float('inf'))}
# }

# selected_range = traffic_ranges[selected_time]

# # ======================
# # STEP 5: Analyze traffic and assign colors
# # ======================
# high, medium, low, unknown = 0, 0, 0, 0
# edge_colors = []

# for u, v, data in G.edges(data=True):
#     level = data["traffic"].get(selected_time)
    
#     # Handle cases where data is None (no traffic data available)
#     if level is None:
#         edge_colors.append("gray")
#         unknown += 1
#     elif level >= selected_range["high"][0]:
#         edge_colors.append("red")
#         high += 1
#     elif selected_range["medium"][0] <= level < selected_range["high"][0]:
#         edge_colors.append("orange")
#         medium += 1
#     elif selected_range["low"][0] <= level < selected_range["medium"][0]:
#         edge_colors.append("green")
#         low += 1
#     else:
#         # Any other case (below the 'low' range), set it to gray
#         edge_colors.append("gray")
#         unknown += 1

# # ======================
# # STEP 6: Print stats
# # ======================
# print(f"\n🚦 Congestion Stats during {selected_time}:")
# print(f"  ✅ Low ({selected_range['low'][0]}–{selected_range['medium'][0]-1}): {low}")
# print(f"  ⚠️ Medium ({selected_range['medium'][0]}–{selected_range['high'][0]-1}): {medium}")
# print(f"  ❌ High (≥ {selected_range['high'][0]}): {high}")
# print(f"  ❓ Unknown/No data: {unknown}")

# # ======================
# # STEP 7: Draw the graph
# # ======================
# pos = nx.spring_layout(G, k=4, iterations=200)
# labels = {node: area_names.get(node, node) for node in G.nodes()}

# plt.figure(figsize=(14, 14))
# nx.draw(G, pos,
#         with_labels=True,
#         labels=labels,
#         node_size=1000,
#         node_color="skyblue",
#         font_size=10,
#         font_weight="bold",
#         edge_color=edge_colors,
#         width=2)

# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# legend_labels = [
#     mpatches.Patch(color='green', label=f"Low (< {selected_range['medium'][0]} veh/h)"),
#     mpatches.Patch(color='orange', label=f"Medium ({selected_range['medium'][0]}–{selected_range['high'][0]-1} veh/h)"),
#     mpatches.Patch(color='red', label=f"High (≥ {selected_range['high'][0]} veh/h)"),
#     mpatches.Patch(color='gray', label="No data / Out of range"),
# ]
# plt.legend(handles=legend_labels, loc="upper left")

# plt.title(f"Traffic Graph - {selected_time}", fontsize=16)
# plt.axis('off')
# plt.show()


import csv
import networkx as nx
import folium

# === STEP 1: Load area names ===
def read_area_names(filename):
    area_names = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            area_names[row["ID"].strip()] = row["Name"].strip()
    return area_names

# === STEP 2: Load traffic data ===
def read_traffic_data(filename):
    traffic = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            road_id = row["RoadID"].strip()
            traffic[road_id] = {
                "Morning": float(row["Morning Peak(veh/h)"]),
                "Afternoon": float(row["Afternoon(veh/h)"]),
                "Evening": float(row["Evening Peak(veh/h)"]),
                "Night": float(row["Night(veh/h)"])
            }
    return traffic

# === STEP 3: Load road data ===
def read_roads(filename, from_col, to_col, dist_col):
    roads = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            roads.append((
                row[from_col].strip(),
                row[to_col].strip(),
                float(row[dist_col].strip())
            ))
    return roads

area_names = read_area_names("area_names.csv")
traffic_data = read_traffic_data("traffic_flow.csv")
existing_roads = read_roads("existing_roads.csv", "FromID", "ToID", "Distance(km)")
new_roads = read_roads("new_roads.csv", "FromID", "ToID", "Distance(km)")
all_roads = existing_roads + new_roads

# === STEP 4: Define traffic levels ===
traffic_ranges = {
    "Morning": {"low": (0, 2000), "medium": (2001, 3000), "high": (3001, float('inf'))},
    "Afternoon": {"low": (0, 1200), "medium": (1201, 1800), "high": (1801, float('inf'))},
    "Evening": {"low": (0, 1500), "medium": (1501, 2400), "high": (2401, float('inf'))},
    "Night": {"low": (0, 300), "medium": (301, 600), "high": (601, float('inf'))}
}

selected_time = "Morning"  # You can make this input if you like
selected_range = traffic_ranges[selected_time]

# === STEP 5: Dummy coordinates (replace with real GPS later) ===
# Just spread nodes in Cairo area randomly
def read_area_locations_with_coords(filename):
    locations = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            area_id = row["ID"].strip()
            lat = float(row["Latitude"].strip())
            lon = float(row["Longitude"].strip())
            locations[area_id] = (lat, lon)
    return locations

area_locations = read_area_locations_with_coords("area_names.csv")


# === STEP 6: Build graph ===
G = nx.Graph()
for from_id, to_id, distance in all_roads:
    road_id_1 = f"{from_id}-{to_id}"
    road_id_2 = f"{to_id}-{from_id}"
    traffic = traffic_data.get(road_id_1) or traffic_data.get(road_id_2)
    G.add_edge(from_id, to_id, weight=distance, traffic=traffic)

# === STEP 7: Create folium map ===
m = folium.Map(location=[30.05, 31.25], zoom_start=11)

# Add markers
for node_id, (lat, lon) in area_locations.items():
    name = area_names.get(node_id, node_id)
    folium.Marker(
        location=[lat, lon],
        popup=name,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Add edges with traffic coloring
for u, v, data in G.edges(data=True):
    lat1, lon1 = area_locations[u]
    lat2, lon2 = area_locations[v]

    color = "gray"  # default
    traffic_level = data["traffic"][selected_time] if data["traffic"] else None

    if traffic_level is not None:
        if traffic_level >= selected_range["high"][0]:
            color = "red"
        elif selected_range["medium"][0] <= traffic_level < selected_range["high"][0]:
            color = "orange"
        elif selected_range["low"][0] <= traffic_level < selected_range["medium"][0]:
            color = "green"

    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color=color,
        weight=4,
        opacity=0.7,
        tooltip=f"Traffic: {traffic_level or 'N/A'} veh/h"
    ).add_to(m)

# === STEP 8: Save map ===
m.save("cairo_traffic_map.html")
print("✅ Map saved as 'cairo_traffic_map.html'. Open it in a browser.")
