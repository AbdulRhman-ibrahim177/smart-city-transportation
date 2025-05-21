import networkx as nx
import folium
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time

class DisjointSet:
    """Implementation of Disjoint Set data structure for Kruskal's algorithm."""
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, node):
        """Find the root parent of a node."""
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, x, y):
        """Union two sets by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

class CairoRoadNetwork:
    """Class to handle Cairo road network operations and MST calculations."""
    
    def __init__(self):
        """Initialize the road network."""
        self.G = nx.Graph()
        self.population_data = {}
        self.critical_facilities = set()
        self.area_names = {}
        self.traffic_data = {}
    
    def load_data(self):
        """Load all required data from CSV files."""
        # Load area names and population
        area_names_df = pd.read_csv("area_names.csv", dtype={"ID": str})
        area_names_df["ID"] = area_names_df["ID"].str.strip()
        self.area_names = dict(zip(area_names_df["ID"], area_names_df["Name"]))
        self.population_data = dict(zip(area_names_df["ID"], area_names_df["Population"]))
        
        # Load traffic data
        traffic_df = pd.read_csv("traffic_flow.csv", dtype={"RoadID": str})
        traffic_df["RoadID"] = traffic_df["RoadID"].str.strip()
        for _, row in traffic_df.iterrows():
            self.traffic_data[row["RoadID"]] = {
                "Morning": float(row["Morning Peak(veh/h)"]),
                "Afternoon": float(row["Afternoon(veh/h)"]),
                "Evening": float(row["Evening Peak(veh/h)"]),
                "Night": float(row["Night(veh/h)"])
            }
        
        # Load road data
        existing_roads = pd.read_csv("existing_roads.csv", dtype={"FromID": str, "ToID": str})
        new_roads = pd.read_csv("new_roads.csv", dtype={"FromID": str, "ToID": str})
        
        # Load facilities
        facilities_df = pd.read_csv("facilities.csv", dtype={"ID": str})
        facilities_df["ID"] = facilities_df["ID"].str.strip()
        for _, row in facilities_df.iterrows():
            self.critical_facilities.add(row["ID"])
        
        # Add nodes with positions
        for _, row in area_names_df.iterrows():
            self.G.add_node(str(row["ID"]).strip(), 
                          pos=(row["Latitude"], row["Longitude"]),
                          name=row["Name"])
        
        # Debug: Print node and edge counts and sample data
        print("Loaded nodes:", len(self.G.nodes()))
        print("Loaded edges:", len(self.G.edges()))
        print("Critical facilities:", self.critical_facilities)
        for node in list(self.G.nodes())[:5]:
            print("Sample node:", node, self.G.nodes[node])
        
        # Debug: Check for missing nodes in edge files
        node_ids = set(self.G.nodes())
        for _, row in existing_roads.iterrows():
            from_id = str(row["FromID"]).strip()
            to_id = str(row["ToID"]).strip()
            if from_id not in node_ids or to_id not in node_ids:
                print(f"Edge references missing node: {from_id} or {to_id}")
        for _, row in new_roads.iterrows():
            from_id = str(row["FromID"]).strip()
            to_id = str(row["ToID"]).strip()
            if from_id not in node_ids or to_id not in node_ids:
                print(f"Edge references missing node: {from_id} or {to_id}")
        
        # Add edges from existing roads
        for _, row in existing_roads.iterrows():
            from_id = str(row["FromID"]).strip()
            to_id = str(row["ToID"]).strip()
            road_id = f"{from_id}-{to_id}"
            traffic_val = None
            if road_id in self.traffic_data:
                traffic_val = self.traffic_data[road_id]
            self.G.add_edge(from_id, to_id, 
                          weight=row["Distance(km)"],
                          distance=row["Distance(km)"],
                          traffic=traffic_val)
        
        # Add edges from new roads
        for _, row in new_roads.iterrows():
            from_id = str(row["FromID"]).strip()
            to_id = str(row["ToID"]).strip()
            self.G.add_edge(from_id, to_id, 
                          weight=row["Distance(km)"],
                          distance=row["Distance(km)"],
                          traffic=None)
    
    def calculate_edge_priority(self, u: str, v: str, data: Dict[str, Any]) -> float:
        """
        Calculate priority score for an edge based on:
        1. Critical facility connection
        2. Population served
        3. Distance
        """
        # Base priority is inverse of distance (shorter distance = higher priority)
        priority = 1.0 / data['weight']
        
        # Boost priority if edge connects to critical facility
        if u in self.critical_facilities or v in self.critical_facilities:
            priority *= 2.0
        
        # Boost priority based on population served
        population_factor = (self.population_data.get(u, 0) + self.population_data.get(v, 0)) / 1000
        priority *= (1 + population_factor)
        
        return priority
    
    def kruskal_mst_with_critical_backbone(self):
        """
        Construct an MST that guarantees all critical facilities are connected first (as a backbone),
        then connects the rest of the network, ensuring a single connected component.
    Returns:
            mst_edges: List of (u, v) tuples representing the MST edges.
        """
        # --- Step 1: Build MST on critical facilities (backbone) ---
        critical_subgraph = self.G.subgraph(self.critical_facilities)
        critical_edges = sorted(critical_subgraph.edges(data=True), key=lambda x: x[2]['weight'])
        ds_critical = DisjointSet(self.critical_facilities)
        backbone_edges = []
        for u, v, data in critical_edges:
            if ds_critical.find(u) != ds_critical.find(v):
                backbone_edges.append((u, v))
                ds_critical.union(u, v)
        # --- Step 2: Build MST for the full graph, skipping backbone edges ---
        all_edges = sorted(self.G.edges(data=True), key=lambda x: x[2]['weight'])
        ds_full = DisjointSet(self.G.nodes())
        # Union the critical backbone in the full disjoint set
        for u, v in backbone_edges:
            ds_full.union(u, v)
        mst_edges = list(backbone_edges)
        for u, v, data in all_edges:
            if (u, v) in mst_edges or (v, u) in mst_edges:
                continue
            if ds_full.find(u) != ds_full.find(v):
                mst_edges.append((u, v))
                ds_full.union(u, v)
        # --- Step 3: Ensure single connected component ---
        # If not all nodes are connected, continue adding lowest-weight edges
        components = set(ds_full.find(n) for n in self.G.nodes())
        if len(components) > 1:
            for u, v, data in all_edges:
                if (u, v) in mst_edges or (v, u) in mst_edges:
                    continue
                if ds_full.find(u) != ds_full.find(v):
                    mst_edges.append((u, v))
                    ds_full.union(u, v)
                if len(set(ds_full.find(n) for n in self.G.nodes())) == 1:
                    break
        return mst_edges

    def detailed_cost_analysis(self, mst_edges):
        """
        Provide a detailed cost-effectiveness analysis for the network.
        Includes construction and maintenance costs for MST and non-MST edges.
        Returns a summary dictionary and a DataFrame for reporting.
        """
        # Example: Assume maintenance cost is 10% of construction cost per edge
        MAINTENANCE_RATE = 0.10
        mst_costs = []
        non_mst_costs = []
        for u, v in self.G.edges():
            edge_data = self.G.get_edge_data(u, v)
            distance = edge_data['weight']
            construction_cost = distance  # You can scale this as needed
            maintenance_cost = construction_cost * MAINTENANCE_RATE
            is_mst = (u, v) in mst_edges or (v, u) in mst_edges
            record = {
                'edge': f"{u}-{v}",
                'distance': distance,
                'construction_cost': construction_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': construction_cost + maintenance_cost,
                'is_mst': is_mst
            }
            if is_mst:
                mst_costs.append(record)
            else:
                non_mst_costs.append(record)
        df = pd.DataFrame(mst_costs + non_mst_costs)
        summary = {
            'mst_construction_cost': sum(r['construction_cost'] for r in mst_costs),
            'mst_maintenance_cost': sum(r['maintenance_cost'] for r in mst_costs),
            'mst_total_cost': sum(r['total_cost'] for r in mst_costs),
            'nonmst_construction_cost': sum(r['construction_cost'] for r in non_mst_costs),
            'nonmst_maintenance_cost': sum(r['maintenance_cost'] for r in non_mst_costs),
            'nonmst_total_cost': sum(r['total_cost'] for r in non_mst_costs),
        }
        return summary, df
    
    def analyze_network(self, mst_edges):
        """Analyze the network and MST."""
        total_population = sum(self.population_data.values())
        total_nodes = len(self.G.nodes())
        total_edges = len(self.G.edges())
        
        # Calculate MST statistics
        mst_total_distance = sum(self.G[u][v]['weight'] for u, v in mst_edges)
        mst_avg_distance = mst_total_distance / len(mst_edges) if mst_edges else 0
        
        # Calculate connectivity metrics
        critical_connected = sum(1 for u, v in mst_edges if u in self.critical_facilities or v in self.critical_facilities)
        total_critical = len(self.critical_facilities)

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'mst_edges': len(mst_edges),
            'total_population': total_population,
            'mst_total_distance': mst_total_distance,
            'mst_avg_distance': mst_avg_distance,
            'critical_connectivity': (critical_connected / total_critical) * 100 if total_critical > 0 else 0
        }
    
    def enhanced_cost_analysis(self, mst_edges):
        """Perform enhanced cost analysis of the network."""
        # Calculate costs for MST edges
        mst_costs = []
        for u, v in mst_edges:
            if u not in self.population_data or v not in self.population_data:
                continue
            distance = self.G[u][v]['weight']
            population = self.population_data[u] + self.population_data[v]
            is_critical = u in self.critical_facilities or v in self.critical_facilities
            
            mst_costs.append({
                'edge': f"{u}-{v}",
                'distance': distance,
                'population': population,
                'is_critical': is_critical,
                'cost_per_km': population / distance if distance > 0 else float('inf')
            })
        
        # Calculate costs for non-MST edges
        non_mst_costs = []
        for u, v in self.G.edges():
            if (u, v) not in mst_edges and (v, u) not in mst_edges:
                if u not in self.population_data or v not in self.population_data:
                    continue
                distance = self.G[u][v]['weight']
                population = self.population_data[u] + self.population_data[v]
                is_critical = u in self.critical_facilities or v in self.critical_facilities
                
                non_mst_costs.append({
                    'edge': f"{u}-{v}",
                    'distance': distance,
                    'population': population,
                    'is_critical': is_critical,
                    'cost_per_km': population / distance if distance > 0 else float('inf')
                })
        
        return {
            'cost_breakdown': mst_costs + non_mst_costs,
            'mst_total_cost': sum(c['cost_per_km'] for c in mst_costs),
            'non_mst_total_cost': sum(c['cost_per_km'] for c in non_mst_costs)
        }
    
    def analyze_neighborhood_connectivity(self, mst_edges):
        """Analyze neighborhood connectivity in the network."""
        # Convert MST edges to a set for faster lookup
        mst_edge_set = set((u, v) if u < v else (v, u) for u, v in mst_edges)
        
        # Analyze connectivity for each neighborhood
        neighborhood_stats = []
        for node in self.G.nodes():
            # Count MST connections
            mst_connections = sum(1 for neighbor in self.G.neighbors(node)
                                if (node, neighbor) in mst_edge_set or (neighbor, node) in mst_edge_set)
            
            # Count total connections
            total_connections = len(list(self.G.neighbors(node)))
            
            # Calculate connectivity ratio
            connectivity_ratio = mst_connections / total_connections if total_connections > 0 else 0
            
            neighborhood_stats.append({
                'node': node,
                'mst_connections': mst_connections,
                'total_connections': total_connections,
                'connectivity_ratio': connectivity_ratio
            })
        
        return {
            'neighborhood_stats': neighborhood_stats,
            'avg_connectivity': sum(s['connectivity_ratio'] for s in neighborhood_stats) / len(neighborhood_stats) if neighborhood_stats else 0
        }
    
    def create_network_map(self, mst_edges):
        """Create a Folium map for the network visualization (lines and icons)."""
        m = folium.Map(location=[30.0444, 31.2357], zoom_start=11)

        # Draw MST edges (yellow)
        for u, v in mst_edges:
            u_pos = self.G.nodes[u].get('pos')
            v_pos = self.G.nodes[v].get('pos')
            if u_pos is None or v_pos is None:
                continue
            folium.PolyLine(
                locations=[u_pos, v_pos],
                color='#FFD600',  # bright yellow
                weight=4,
                opacity=0.8,
                popup=f"MST Edge: {self.G.nodes[u]['name']} - {self.G.nodes[v]['name']}"
            ).add_to(m)

        # Draw other edges (orange)
        for u, v in self.G.edges():
            if (u, v) in mst_edges or (v, u) in mst_edges:
                continue
            u_pos = self.G.nodes[u].get('pos')
            v_pos = self.G.nodes[v].get('pos')
            if u_pos is None or v_pos is None:
                continue
            folium.PolyLine(
                locations=[u_pos, v_pos],
                color='#FF9100',  # bright orange
                weight=2,
                opacity=0.6,
                popup=f"Edge: {self.G.nodes[u]['name']} - {self.G.nodes[v]['name']}"
            ).add_to(m)

        # Draw nodes: red star for critical, red pin for others
        for node in self.G.nodes():
            pos = self.G.nodes[node].get('pos')
            if pos is None:
                continue
            name = self.G.nodes[node].get('name', node)
            if node in self.critical_facilities:
                # Red star for critical facility (FontAwesome)
                folium.Marker(
                    location=pos,
                    icon=folium.Icon(color='red', icon='star', prefix='fa'),
                    popup=f"⭐ Critical Facility: {name}"
                ).add_to(m)
            else:
                # Red pin for population center (FontAwesome)
                folium.Marker(
                    location=pos,
                    icon=folium.Icon(color='red', icon='map-marker', prefix='fa'),
                    popup=f"📍 Population center: {name}"
        ).add_to(m)

        return m, self.G

def main():
    """Main function to demonstrate the usage of CairoRoadNetwork class."""
    # Create and initialize the road network
    network = CairoRoadNetwork()
    network.load_data()
    
    # Find MST
    mst_edges = network.kruskal_mst_with_critical_backbone()
    
    # Get network analysis
    basic_analysis = network.analyze_network(mst_edges)
    enhanced_analysis = network.enhanced_cost_analysis(mst_edges)
    connectivity_analysis = network.analyze_neighborhood_connectivity(mst_edges)
    
    # Create and save the map
    m, _ = network.create_network_map(mst_edges)
    m.save("cairo_road_network_mst.html")
    
    print("✅ Map saved as 'cairo_road_network_mst.html'. Open it in a browser.")
    print("\nNetwork Analysis Results:")
    print(f"Total Nodes: {basic_analysis['total_nodes']}")
    print(f"Total Edges: {basic_analysis['total_edges']}")
    print(f"MST Edges: {basic_analysis['mst_edges']}")
    print(f"Total Population: {basic_analysis['total_population']:,}")
    print(f"MST Total Distance: {basic_analysis['mst_total_distance']:.2f} km")
    print(f"MST Average Distance: {basic_analysis['mst_avg_distance']:.2f} km")
    print(f"Critical Facility Connectivity: {basic_analysis['critical_connectivity']:.1f}%")

if __name__ == "__main__":
    main() 