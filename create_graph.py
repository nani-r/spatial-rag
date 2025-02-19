
import overpy
from geopy.distance import geodesic
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef, BNode
from geopy.distance import geodesic

def fetch_cities_in_country(country_name):
    """
    Fetches approximately 300 primary cities in a specified country using Overpass API.
    
    Args:
        country_name (str): Name of the country.
    
    Returns:
        list: A list of dictionaries containing city names and their coordinates.
    """
    api = overpy.Overpass()


    query = f"""
    [out:json];
    area["name"="{country_name}"][admin_level=2];
    node[place="city"](area);
    
    out;
    """
    result = api.query(query)
    #node[place~"city|town"]["population"](if: t["population"] >= 5000)(area);

    # Extract city names and coordinates
    primary_cities = []
    for node in result.nodes:
        primary_cities.append({"name": node.tags.get("name", "unknown"), "lat": node.lat, "lon": node.lon})
    print("length/numcitites: ", len(primary_cities))
    return primary_cities


def create_city_graph(cities):
    """
    Generates a graph of cities with distances between them.
    
    Args:
        cities (list): List of dictionaries containing city names and coordinates.
    
    Returns:
        pd.DataFrame: A DataFrame with city1, city2, and distance.
    """
    connections = []
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j: 
                dist = geodesic((city1["lat"], city1["lon"]), (city2["lat"], city2["lon"])).kilometers
                connections.append((city1["name"], city2["name"], round(dist)))
    

    df = pd.DataFrame(connections, columns=["City1", "City2", "Distance (km)"])
    return connections



def create_rdf_city_graph(cities, output_filename="city_graph.ttl"):
    """
    Generates an RDF graph representing cities and their distances using blank nodes.
    
    Args:
        cities (list): List of dictionaries containing city names and coordinates.
    
    Returns:
        rdflib.Graph: RDF graph containing city relationships and distances.
    """
    g = Graph()
    EX = Namespace("http://example.org/cities#")
    
    for city in cities:
        city_uri = URIRef(EX + city["name"].replace(" ", "_"))
        g.add((city_uri, RDF.type, EX.City))
    
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                dist = geodesic((city1["lat"], city1["lon"]), (city2["lat"], city2["lon"])).kilometers
                city1_uri = URIRef(EX + city1["name"].replace(" ", "_"))
                city2_uri = URIRef(EX + city2["name"].replace(" ", "_"))
                
     
                distance_relation = BNode()
                g.add((city1_uri, EX.distanceTo, distance_relation))
                g.add((distance_relation, EX.destination, city2_uri))
                g.add((distance_relation, EX.distance, Literal(round(dist))))
    
    g.serialize(destination=output_filename, format="turtle")
    return g
