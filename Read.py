
import csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ======================
# STEP 0: Read area names from CSV
# ======================
def read_area_names(filename):
    area_names = {}
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            area_id = row["ID"].strip()
            name = row["Name"].strip()
            area_names[area_id] = name
    return area_names

area_names = read_area_names("area_names.csv")

# ======================
# STEP 1: Read roads and traffic data
# ======================
def read_roads_from_csv(filename, from_col, to_col, distance_col):
    roads = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [header.strip() for header in reader.fieldnames]
        for row in reader:
            from_id = row[from_col].strip()
            to_id = row[to_col].strip()
            distance = float(row[distance_col].strip())
            roads.append((from_id, to_id, distance))
    return roads

def read_traffic_data(filename):
    traffic_data = {}
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [header.strip() for header in reader.fieldnames]
        for row in reader:
            road_id = row["RoadID"].strip()
            try:
                traffic_data[road_id] = {
                    "Morning": float(row["Morning Peak(veh/h)"].strip()),
                    "Afternoon": float(row["Afternoon(veh/h)"].strip()),
                    "Evening": float(row["Evening Peak(veh/h)"].strip()),
                    "Night": float(row["Night(veh/h)"].strip()),
                }
            except ValueError:
                print(f"⚠️ Error reading traffic data for road {road_id}, skipping.")
                traffic_data[road_id] = {"Morning": None, "Afternoon": None, "Evening": None, "Night": None}
    return traffic_data

existing_roads = read_roads_from_csv("existing_roads.csv", "FromID", "ToID", "Distance(km)")
new_roads = read_roads_from_csv("new_roads.csv", "FromID", "ToID", "Distance(km)")
traffic_data = read_traffic_data("Traffic-flow.csv")
all_roads = existing_roads + new_roads

# ======================
# STEP 2: Ask for time of day
# ======================
print("⏰ Available times: Morning, Afternoon, Evening, Night")
selected_time = input("Enter desired time for traffic visualization: ").capitalize()

if selected_time not in ["Morning", "Afternoon", "Evening", "Night"]:
    print("❌ Invalid time! Using 'Morning' as default.")
    selected_time = "Morning"

# ======================
# STEP 3: Build Graph
# ======================
G = nx.Graph()

for from_id, to_id, distance in all_roads:
    road_id = f"{from_id}-{to_id}" if f"{from_id}-{to_id}" in traffic_data else f"{to_id}-{from_id}"
    traffic = traffic_data.get(road_id, {"Morning": None, "Afternoon": None, "Evening": None, "Night": None})
    G.add_edge(from_id, to_id, weight=distance, traffic=traffic)

# ======================
# STEP 4: Define traffic ranges
# ======================
traffic_ranges = {
    "Morning": {"low": (0, 2000), "medium": (2001, 3000), "high": (3001, float('inf'))},
    "Afternoon": {"low": (0, 1200), "medium": (1201, 1800), "high": (1801, float('inf'))},
    "Evening": {"low": (0, 1500), "medium": (1501, 2400), "high": (2401, float('inf'))},
    "Night": {"low": (0, 300), "medium": (301, 600), "high": (601, float('inf'))}
}

selected_range = traffic_ranges[selected_time]

# ======================
# STEP 5: Analyze traffic and assign colors
# ======================
high, medium, low, unknown = 0, 0, 0, 0
edge_colors = []

for u, v, data in G.edges(data=True):
    level = data["traffic"].get(selected_time)
    
    # Handle cases where data is None (no traffic data available)
    if level is None:
        edge_colors.append("gray")
        unknown += 1
    elif level >= selected_range["high"][0]:
        edge_colors.append("red")
        high += 1
    elif selected_range["medium"][0] <= level < selected_range["high"][0]:
        edge_colors.append("orange")
        medium += 1
    elif selected_range["low"][0] <= level < selected_range["medium"][0]:
        edge_colors.append("green")
        low += 1
    else:
        # Any other case (below the 'low' range), set it to gray
        edge_colors.append("gray")
        unknown += 1

# ======================
# STEP 6: Print stats
# ======================
print(f"\n🚦 Congestion Stats during {selected_time}:")
print(f"  ✅ Low ({selected_range['low'][0]}–{selected_range['medium'][0]-1}): {low}")
print(f"  ⚠️ Medium ({selected_range['medium'][0]}–{selected_range['high'][0]-1}): {medium}")
print(f"  ❌ High (≥ {selected_range['high'][0]}): {high}")
print(f"  ❓ Unknown/No data: {unknown}")

# ======================
# STEP 7: Draw the graph
# ======================
pos = nx.spring_layout(G, k=4, iterations=200)
labels = {node: area_names.get(node, node) for node in G.nodes()}

plt.figure(figsize=(14, 14))
nx.draw(G, pos,
        with_labels=True,
        labels=labels,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color=edge_colors,
        width=2)

edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

legend_labels = [
    mpatches.Patch(color='green', label=f"Low (< {selected_range['medium'][0]} veh/h)"),
    mpatches.Patch(color='orange', label=f"Medium ({selected_range['medium'][0]}–{selected_range['high'][0]-1} veh/h)"),
    mpatches.Patch(color='red', label=f"High (≥ {selected_range['high'][0]} veh/h)"),
    mpatches.Patch(color='gray', label="No data / Out of range"),
]
plt.legend(handles=legend_labels, loc="upper left")

plt.title(f"Traffic Graph - {selected_time}", fontsize=16)
plt.axis('off')
plt.show()
