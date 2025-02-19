import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import time

os.environ["OPENAI_API_KEY"] = "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"  


llm = ChatOpenAI(temperature=0, model_name="gpt-4")


prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    Given the following question, provide a direct answer. Only return the distance (km) or the city name (text) as requested in the question. Do not include any additional information including units.
    Do not provide any additional context except the answer.
    Question:
    {question}

    Answer:
    """
)


chain = prompt | llm

# Function to query the LLM with the question and get an answer
def query_llm(question_text):
    """Generates a response from the LLM for a given question."""
    return chain.invoke({"question": question_text}).content.strip()


with open("city_questions.json", "r") as f:
    questions = json.load(f)

start_time = time.time()


formatted_answers = []


for q in questions:
    for level in ["easy", "medium", "hard"]:
        question_text = q[level]["question"]
        answer = query_llm(question_text)
        formatted_answers.append({
            level: {
                "question": question_text,
                "answer": answer
            }
        })
end_time = time.time()
execution_time = end_time - start_time


result = {
    "answers": formatted_answers,
    "execution_time_seconds": execution_time
}

with open("llm_answers.json", "w") as f:
    json.dump(result, f, indent=4)

print("Answers saved to llm_answers.json")
