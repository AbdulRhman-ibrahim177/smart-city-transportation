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

# === STEP 4: Traffic level thresholds ===
traffic_ranges = {
    "Morning": {"low": (0, 2000), "medium": (2001, 3000), "high": (3001, float('inf'))},
    "Afternoon": {"low": (0, 1200), "medium": (1201, 1800), "high": (1801, float('inf'))},
    "Evening": {"low": (0, 1500), "medium": (1501, 2400), "high": (2401, float('inf'))},
    "Night": {"low": (0, 300), "medium": (301, 600), "high": (601, float('inf'))}
}
selected_time = "Morning"
selected_range = traffic_ranges[selected_time]

# === STEP 5: Load area coordinates ===
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

# === STEP 6: Load facilities ===
def read_facilities(filename):
    facilities = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facility_id = row["ID"].strip()
            facilities[facility_id] = {
                "name": row["Name"].strip(),
                "type": row["Type"].strip(),
                "location": (float(row["Y-coordinate"]), float(row["X-coordinate"]))  # lat, lon
            }
    return facilities

# === STEP 7: Load bus routes ===
def read_bus_routes(filename):
    routes = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stops = row["Stops(comma-separated IDs)"].replace('"', '').split(',')
            stops = [s.strip() for s in stops]
            routes.append({
                "id": row["RouteID"].strip(),
                "stops": stops,
                "buses": int(row["Buses Assigned"]),
                "passengers": int(row["Daily Passengers"])
            })
    return routes

# === STEP 8: Load metro routes ===
def read_metro_routes(filename):
    metro_lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations = row["Stations(comma-separated IDs)"].replace('"', '').split(',')
            stations = [s.strip() for s in stations]
            metro_lines.append({
                "id": row["LineID"].strip(),
                "name": row["Name"].strip(),
                "stations": stations,
                "passengers": int(row["Daily Passengers"])
            })
    return metro_lines

# === STEP 9: Load data ===
area_names = read_area_names("area_names.csv")
traffic_data = read_traffic_data("traffic_flow.csv")
existing_roads = read_roads("existing_roads.csv", "FromID", "ToID", "Distance(km)")
new_roads = read_roads("new_roads.csv", "FromID", "ToID", "Distance(km)")
area_locations = read_area_locations_with_coords("area_names.csv")
facilities = read_facilities("facilities.csv")
bus_routes = read_bus_routes("bus_routes.csv")
metro_routes = read_metro_routes("metro_lines.csv")
all_roads = existing_roads + new_roads

# === STEP 10: Build graph ===
G = nx.Graph()
for from_id, to_id, distance in all_roads:
    road_id_1 = f"{from_id}-{to_id}"
    road_id_2 = f"{to_id}-{from_id}"
    traffic = traffic_data.get(road_id_1) or traffic_data.get(road_id_2)
    G.add_edge(from_id, to_id, weight=distance, traffic=traffic)

# === STEP 11: Create map ===
m = folium.Map(location=[30.05, 31.25], zoom_start=11)

# Add area markers
for node_id, (lat, lon) in area_locations.items():
    name = area_names.get(node_id, node_id)
    folium.Marker(
        location=[lat, lon],
        popup=name,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Add facility markers
for fid, info in facilities.items():
    folium.Marker(
        location=info["location"],
        popup=f"{info['name']} ({info['type']})",
        icon=folium.Icon(color="purple", icon="star")
    ).add_to(m)

# Add traffic edges
for u, v, data in G.edges(data=True):
    lat1, lon1 = area_locations[u]
    lat2, lon2 = area_locations[v]
    color = "gray"
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

# Add bus routes
for route in bus_routes:
    path = []
    for stop in route["stops"]:
        loc = area_locations.get(stop)
        if loc:
            path.append(loc)
    if len(path) >= 2:
        folium.PolyLine(
            path,
            color="blue",
            weight=3,
            opacity=0.5,
            tooltip=f"Bus Route {route['id']} | {route['passengers']} passengers/day"
        ).add_to(m)

# Add metro routes
for line in metro_routes:
    path = []
    for station in line["stations"]:
        loc = area_locations.get(station) or facilities.get(station, {}).get("location")
        if loc:
            path.append(loc)
    if len(path) >= 2:
        folium.PolyLine(
            path,
            color="darkgreen",
            weight=4,
            opacity=0.6,
            tooltip=f"Metro {line['name']} | {line['passengers']} passengers/day"
        ).add_to(m)

# === STEP 12: Save map ===
m.save("cairo_traffic_map.html")
print("✅ Map saved as 'cairo_traffic_map.html'. Open it in a browser.")
