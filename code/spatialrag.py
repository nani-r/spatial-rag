import os
import json
import networkx as nx
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import warnings
import time
from create_graph import create_city_graph, fetch_cities_in_country

os.environ["OPENAI_API_KEY"] = "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"  


warnings.filterwarnings("ignore")


def create_and_embed_graph(cities):
    """
    Creates a graph from city connections and embeds it using OpenAIEmbeddings.

    Args:
        cities: A list of cities.

    Returns:
        A tuple containing the graph (nx.Graph) and the FAISS vector store.
    """
    G = nx.Graph()
    
    city_connections = create_city_graph(cities)

    for city1, city2, distance in city_connections:
        G.add_edge(city1, city2, weight=distance)

    # Embed graph data with city names and routes
    graph_data = []
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        graph_data.append({
            "text": f"Distance between {u} and {v} is {weight} km",
            "metadata": {"city1": u, "city2": v, "distance": weight}
        })
        graph_data.append({
            "text": f"Distance between {v} and {u} is {weight} km",
            "metadata": {"city1": v, "city2": u, "distance": weight}
        })

    # Embed the text data with OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    graph_store = FAISS.from_texts([item["text"] for item in graph_data], embeddings)
    
    # Create a retriever from the FAISS vector store
    retriever = graph_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    return G, retriever


def ask_graph_question(question, retriever):
    """
    Queries the graph and asks the LLM a question.

    Args:
        question: The question to ask.
        retriever: The FAISS retriever object.

    Returns:
        The LLM's response.
    """

    # Retrieve relevant graph data
    docs = retriever.get_relevant_documents(question)
    graph_context = "\n".join([doc.page_content for doc in docs])


    prompt = custom_graph_prompt(question, graph_context)

    # QA Chain setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        return_source_documents=True,
    )


    response = qa_chain.invoke({"query": prompt})
    return response["result"]


def custom_graph_prompt(question, graph_context):
    """
    Creates a concise prompt for the graph-based question.

    Args:
        question: The question to ask.
        graph_context: The relevant graph context.

    Returns:
        The formatted prompt string.
    """
    return f"""
    You are an assistant with access to a graph database containing distances between cities. 

    Your task is to extract and return the distance between the cities mentioned in the question. 

    Given the following question, provide a direct answer. Only return the distance (km) or the city name (text) as requested in the question. Do not include any additional information including units.
    
    Question:
    {question}
    Context:
    {graph_context}
    """


def process_questions_and_dump_answers(cities, questions):
    """
    Processes all questions, retrieves answers using the RAG approach, and dumps the answers into a JSON file.

    Args:
        cities: List of cities to be included in the graph.
        questions: List of questions to answer.

    Returns:
        None
    """
    start_time = time.time()
    # Create graph and retriever
    _, retriever = create_and_embed_graph(cities)


    answers = []


    for q in questions:
        for level in ["easy", "medium", "hard"]:
            question_text = q[level]["question"]
            answer = ask_graph_question(question_text, retriever)
            answers.append({
                level: {
                    "question": question_text,
                    "answer": answer
                }
            })

    end_time = time.time()  
    execution_time = end_time - start_time
    result = {
        "answers": answers,
        "execution_time_seconds": execution_time
    }

    with open("spatialrag_answers.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Answers have been saved to spatialrag_answers.json")

def load_questions_from_json(file_path):
    """
    Loads questions from a given JSON file.

    Args:
        file_path: The path to the JSON file containing the questions.

    Returns:
        A list of questions.
    """
    with open(file_path, "r") as file:
        questions = json.load(file)
    return questions


cities = fetch_cities_in_country("Australia")
questions = load_questions_from_json("city_questions.json")
process_questions_and_dump_answers(cities, questions)
