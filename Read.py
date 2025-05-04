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
