# import csv
# import networkx as nx

# class TransitOptimizer:
#     def __init__(self, graph, area_locations, traffic_data, area_names, selected_time="Morning"):
#         self.graph = graph
#         self.locations = area_locations
#         self.traffic = traffic_data
#         self.names = area_names
#         self.selected_time = selected_time
#         self.routes = {}

#     def load_bus_routes(self, filename):
#         self.routes["bus"] = []
#         with open(filename, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 route = {
#                     "RouteID": row["RouteID"].strip(),
#                     "Stops": row["Stops(comma-separated IDs)"].replace('"', "").strip().split(','),
#                     "Buses": int(row["Buses Assigned"]),
#                     "Passengers": int(row["Daily Passengers"])
#                 }
#                 self.routes["bus"].append(route)

#     def load_metro_lines(self, filename):
#         self.routes["metro"] = []
#         with open(filename, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 route = {
#                     "RouteID": row["LineID"].strip(),
#                     "Name": row["Name"].strip(),
#                     "Stops": row["Stations(comma-separated IDs)"].replace('"', "").strip().split('/'),
#                     "Passengers": int(row["Daily Passengers"])
#                 }
#                 self.routes["metro"].append(route)

#     def estimate_route_time(self, stops):
#         total_time = 0.0
#         for i in range(len(stops) - 1):
#             u, v = stops[i].strip(), stops[i+1].strip()
#             if self.graph.has_edge(u, v):
#                 edge = self.graph[u][v]
#                 dist = edge['weight']
#                 traffic_level = edge.get('traffic', {}).get(self.selected_time, 1000)
#                 avg_speed = 25 if traffic_level < 2000 else 15  # example logic
#                 travel_time = dist / avg_speed * 60
#                 total_time += travel_time
#             else:
#                 print(f"⚠️ No direct connection between {u} and {v}")
#         return round(total_time, 2)

#     def analyze_all_routes(self):
#         print("📊 Estimated Travel Times:")
#         for mode in self.routes:
#             for route in self.routes[mode]:
#                 time_min = self.estimate_route_time(route["Stops"])
#                 route_id = route["RouteID"]
#                 print(f" - {mode.title()} {route_id} → Estimated Time: {time_min} min")

#         self.find_transfer_points()

#     def find_transfer_points(self):
#         print("\n🔁 Transfer Points Between Routes:")
#         stop_counter = {}
#         for mode in self.routes:
#             for route in self.routes[mode]:
#                 for stop in route["Stops"]:
#                     stop = stop.strip()
#                     stop_counter[stop] = stop_counter.get(stop, 0) + 1

#         for stop, count in stop_counter.items():
#             if count > 1:
#                 name = self.names.get(stop, stop)
#                 print(f" - {name} (used in {count} routes)")



def optimize_vehicle_allocation(lines, total_vehicles):
    """
    lines: List of dicts like {'id': 'B1', 'passengers': 35000, 'assigned': 25}
    total_vehicles: Total number of available vehicles to allocate
    Returns a dict of optimized allocation per line.
    """
    n = len(lines)
    dp = [[0] * (total_vehicles + 1) for _ in range(n + 1)]

    # Build DP table
    for i in range(1, n + 1):
        for v in range(total_vehicles + 1):
            line = lines[i - 1]
            base_score = line['passengers']  # benefit of allocating at least 1 bus
            vehicles_needed = line['assigned']  # estimated need

            if v >= vehicles_needed:
                dp[i][v] = max(
                    dp[i - 1][v],  # skip this line
                    dp[i - 1][v - vehicles_needed] + base_score  # assign vehicles
                )
            else:
                dp[i][v] = dp[i - 1][v]

    # Backtrack to get the optimal assignment
    result = {}
    v = total_vehicles
    for i in range(n, 0, -1):
        if dp[i][v] != dp[i - 1][v]:
            line = lines[i - 1]
            result[line['id']] = line['assigned']
            v -= line['assigned']
        else:
            result[lines[i - 1]['id']] = 0

    return result
import csv

def read_bus_data(filename):
    bus_lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            bus_lines.append({
                'id': row['RouteID'].strip(),
                'passengers': int(row['Daily Passengers'].strip()),
                'assigned': int(row['Buses Assigned'].strip())
            })
    return bus_lines
def read_metro_data(filename):
    metro_lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            metro_lines.append({
                'id': row['LineID'].strip(),
                'passengers': int(row['Daily Passengers'].strip()),
                'assigned': len(row['Stations(comma-separated IDs)'].strip().split('/'))  # تقريب لعدد القطارات المطلوبة
            })
    return metro_lines
bus_lines = read_bus_data('bus_routes.csv')
metro_lines = read_metro_data('metro_lines.csv')

all_lines = bus_lines + metro_lines
optimized = optimize_vehicle_allocation(all_lines, total_vehicles=150)
print(optimized)
