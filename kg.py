import os
import json
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from create_graph import fetch_cities_in_country, create_rdf_city_graph
from decimal import Decimal
import time


os.environ["OPENAI_API_KEY"] = "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" 
def decimal_default(obj):
    """Default JSON serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def fetch_cities_in_country_cached(country, cache_file="cities.json"):
    """Fetch city data, but cache it locally to avoid repeated API calls."""
    if os.path.exists(cache_file):
        try:
            print("Loading cached city data...")
            with open(cache_file, "r") as f:
                cities = json.load(f)
            return cities
        except json.JSONDecodeError:
            print(f"Error reading JSON from {cache_file}, fetching fresh data.")
    

    cities = fetch_cities_in_country(country)


    with open(cache_file, "w") as f:
        json.dump(cities, f, indent=4, default=decimal_default)

    return cities

cities =fetch_cities_in_country("Australia")
graph = create_rdf_city_graph(cities)


llm = ChatOpenAI(temperature=0, model_name="gpt-4")

prompt = PromptTemplate(
    input_variables=["question"],
    template="""Convert the following natural language question into a SPARQL query for an RDF graph.

The RDF graph follows this structure:
- Cities are identified by URIs in the namespace <http://example.org/cities#>.
- The predicate `ns1:distanceTo` represents relationships between cities.
- The predicate `ns1:distance` represents the numeric distance between cities.

Example RDF:
    ns1:Adelaide a ns1:City ;
    ns1:distanceTo [ ns1:destination ns1:Perth ;
            ns1:distance 2135 ],
        [ ns1:destination ns1:Launceston ;
            ns1:distance 1039 ],
        [ ns1:destination ns1:Cairns ;
            ns1:distance 2119 ],
        [ ns1:destination ns1:Ipswich ;
            ns1:distance 1571 ],
        [ ns1:destination ns1:Mount_Isa ;

Generate a SPARQL query to answer the question. The result should either be a distance or a city name depending on the question.

Note: If there are spaces in city names, you will have to replace them with underscores (_)


ONLY applies for the the hard questions: calculate and store the distance (d) between city1 and city2 in the question.
Next, compare all the distances from the city3 in the question. Find and return the name of city that is the closest distance to (d).

### Expected SPARQL Query:
PREFIX ns1: <http://example.org/cities#>


### Question:
{question}

SPARQL Query:"""
)


chain = prompt | llm | RunnableLambda(lambda output: output)


with open("city_questions.json", "r") as f:
    questions = json.load(f)


def execute_query(question_text):
    """Generates and executes a SPARQL query for a given question, handling errors."""
    try:
        sparql_query = chain.invoke({"question": question_text}).content
        results = graph.query(sparql_query)
        return [row[0] for row in results] if results else None
    except Exception:
        return None 

start_time = time.time()

for q in questions:
    for level in ["easy", "medium", "hard"]:
        question_text = q[level]["question"]
        q[level]["answer"] = execute_query(question_text)

end_time = time.time()
execution_time = end_time - start_time

result = {
    "answers": questions,
    "execution_time_seconds": execution_time
}


with open("kg_answers.json", "w") as f:
    json.dump(result, f, indent=4)

print("Answers saved to kg_answers.json")
