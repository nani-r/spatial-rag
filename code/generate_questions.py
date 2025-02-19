import json
import random
from geopy.distance import geodesic
from create_graph import fetch_cities_in_country, create_rdf_city_graph
from rdflib import Graph, Namespace


def generate_questions(cities, num_questions=20):
    """
    Generates a series of easy, medium, and hard questions based on city distances.
    
    Args:
        cities (list): List of dictionaries containing city names and coordinates.
        num_questions (int): Number of question sets to generate.
    
    Returns:
        list: A list of dictionaries containing easy, medium, and hard questions.
    """
    questions = []
    
    for _ in range(num_questions):
        city1, city2, city3 = random.sample(cities, 3)
        
        question_set = {
            "easy": f"What is the distance between {city1['name']} and {city2['name']}?",
            "medium": f"What's the distance between {city1['name']} and its closest city?",
            "hard": f"The distance from {city1['name']} to {city2['name']} is similar to the distance from {city3['name']} to what other city or town?"
        }
        
        questions.append(question_set)
    
    return questions

def get_distance_from_rdf(graph, city1, city2):
    """
    Queries the RDF graph to find the distance between two cities.
    
    Args:
        graph (rdflib.Graph): RDF graph containing city relationships and distances.
        city1 (str): Name of the first city.
        city2 (str): Name of the second city.
    
    Returns:
        int: Distance between the cities or None if not found.
    """
    EX = Namespace("http://example.org/cities#")
    query = f"""
    PREFIX ns1: <http://example.org/cities#>
    SELECT ?distance WHERE {{
        ns1:{city1.replace(' ', '_')} ns1:distanceTo [ ns1:destination ns1:{city2.replace(' ', '_')} ; ns1:distance ?distance ] .
    }}
    """
    
    result = graph.query(query)
    
    for row in result:
        return int(row["distance"])
    return None

def get_closest_city_distance(graph, city1):
    """
    Finds the closest city to the given city in the RDF graph.

    Args:
        graph (rdflib.Graph): RDF graph containing city distances.
        city1 (str): Name of the city.

    Returns:
        tuple: (Closest city name, Distance) or (None, None) if no match is found.
    """
    EX = Namespace("http://example.org/cities#")
    query = f"""
    PREFIX ns1: <http://example.org/cities#>
    SELECT ?destination ?distance WHERE {{
        ns1:{city1.replace(' ', '_')} ns1:distanceTo [ ns1:destination ?destination ; ns1:distance ?distance ] .
    }}
    ORDER BY ?distance
    LIMIT 1
    """
    
    result = graph.query(query)
    for row in result:
        return str(row["destination"].split("#")[-1]), int(row["distance"])
    
    return None, None

def find_closest_matching_distance(graph, city3, target_distance):
    """
    Finds a city with the closest absolute distance to `target_distance` from `city3`.

    Args:
        graph (rdflib.Graph): RDF graph containing city distances.
        city3 (str): The city to compare distances from.
        target_distance (int): The reference distance.

    Returns:
        tuple: (Best matching city name, Distance) or (None, None) if no match is found.
    """
    EX = Namespace("http://example.org/cities#")
    query = f"""
    PREFIX ns1: <http://example.org/cities#>
    SELECT ?destination ?distance WHERE {{
        ns1:{city3.replace(' ', '_')} ns1:distanceTo [ ns1:destination ?destination ; ns1:distance ?distance ] .
    }}
    """
    
    result = graph.query(query)
    best_city = None
    best_distance = None
    min_diff = float("inf")

    for row in result:
        city_name = str(row["destination"].split("#")[-1])
        distance = int(row["distance"])
        diff = abs(distance - target_distance)
        
        if diff < min_diff:
            min_diff = diff
            best_city = city_name
            best_distance = distance
    
    return best_city, best_distance

def populate_answers(questions, rdf_graph):
    """
    Populates the answers in the questions list by querying the RDF graph.

    Args:
        questions (list): List of question dictionaries.
        rdf_graph (rdflib.Graph): RDF graph containing city relationships and distances.

    Returns:
        list: Updated questions list with answers.
    """
    for q in questions:
        if "easy" in q:
            city1, city2 = q["easy"].split(" between ")[1].split(" and ")
            city2 = city2.rstrip("?")  
            q["easy"] = {
                "question": q["easy"],
                "answer": get_distance_from_rdf(rdf_graph, city1, city2)
            }

        if "medium" in q:
            city1 = q["medium"].split(" between ")[1].split(" and ")[0]
            q["medium"] = {
                "question": q["medium"],
                "answer": get_closest_city_distance(rdf_graph, city1)[1]
            }

        if "hard" in q:
            city1, city2 = q["hard"].split(" from ")[1].split(" to ")[0], q["hard"].split(" to ")[1].split(" is ")[0]
            target_distance = get_distance_from_rdf(rdf_graph, city1, city2)
            city3 = q["hard"].split(" is similar to the distance from ")[1].split(" to ")[0]
            q["hard"] = {
                "question": q["hard"],
                "answer": find_closest_matching_distance(rdf_graph, city3 ,target_distance)[0]
            }

    return questions


# Fetch cities
cities = fetch_cities_in_country("Australia")
questions = generate_questions(cities)


# Load RDF graph
graph = create_rdf_city_graph(cities)
questions = populate_answers(questions, graph)


# Save questions to a JSON file
with open("city_questions.json", "w") as f:
    json.dump(questions, f, indent=4)

print("Generated questions saved to city_questions.json")
