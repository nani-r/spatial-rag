import json
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.ticker import MaxNLocator
def extract_numeric_answer(answer):
    """Extracts a numerical distance from various answer formats."""
    if isinstance(answer, list) and answer:
        answer = answer[0]  
    
    if isinstance(answer, str):
        numbers = re.findall(r"\d+", answer) 
        return int(numbers[0]) if numbers else None

    if isinstance(answer, (int, float)):
        return int(answer) 

    return None  
def load_json(filename):
    """Loads JSON data from a file."""
    with open(filename, "r") as f:
        return json.load(f)

def extract_answers(data):
    """Extracts medium difficulty answers and converts them to numbers."""
    answers = []
    
    if isinstance(data, list):
        source_data = data
    elif isinstance(data, dict):
        source_data = data.get("answers", [])
    else:
        return answers

    for item in source_data:
        if isinstance(item, dict) and "medium" in item and "answer" in item["medium"]:
            num_answer = extract_numeric_answer(item["medium"]["answer"])
            answers.append(num_answer if num_answer is not None else np.nan)
    
    return answers


plain_llm = load_json("llm_answers.json")
spatial_rag = load_json("spatialrag_answers.json")
kg_answers = load_json("kg_answers.json")
city_questions = load_json("city_questions.json")


plain_answers = extract_answers(plain_llm)
spatial_answers = extract_answers(spatial_rag)
kg_answers = extract_answers(kg_answers)
perfect_answers = extract_answers(city_questions)


max_length = max(len(plain_answers), len(spatial_answers), len(kg_answers), len(perfect_answers))

def pad_list(lst, length):
    """Pads a list with NaNs to match a specified length."""
    return lst + [np.nan] * (length - len(lst))

plain_answers = pad_list(plain_answers, max_length)
spatial_answers = pad_list(spatial_answers, max_length)
kg_answers = pad_list(kg_answers, max_length)
perfect_answers = pad_list(perfect_answers, max_length)


def compute_residuals(model_answers, perfect_answers):
    residuals = []
    for model_answer, perfect_answer in zip(model_answers, perfect_answers):
        if np.isnan(model_answer) or np.isnan(perfect_answer):
            residuals.append(np.nan)
        else:
            residuals.append(abs(model_answer - perfect_answer))
    return residuals

plain_residuals = compute_residuals(plain_answers, perfect_answers)
spatial_residuals = compute_residuals(spatial_answers, perfect_answers)
kg_residuals = compute_residuals(kg_answers, perfect_answers)


bins = [0, 100, 200, 300, 400, 500, 600, 700]


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)


titles = [
    "GPT-4",
    "SpatialRAG - Vector Similarity",
    "SpatialRAG - SPARQL"
]

residual_lists = [plain_residuals, spatial_residuals, kg_residuals]


for ax, residuals, title in zip(axes, residual_lists, titles):
    residuals = np.array(residuals)
    nan_mask = np.isnan(residuals)
    residuals[nan_mask] = bins[-1] 
    
    ax.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Residual Error (km)")
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(bins)
    xticklabels = [str(b) for b in bins[:-1]] + [">" + str(bins[-1])] 
    ax.set_xticklabels(xticklabels)  

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


axes[0].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("residual_histograms.png", dpi=300)

plt.show()



def calculate_mse(residuals, perfect_answers):
    """Calculates the Mean Squared Error, ignoring NaNs."""
    valid_residuals = []
    nan_count = 0

    for res, perfect in zip(residuals, perfect_answers):
        if np.isnan(res) or np.isnan(perfect):
            nan_count += 1
        else:
            valid_residuals.append(res ** 2)

    if valid_residuals:
        mse = np.mean(valid_residuals)
    else:
        mse = np.nan  

    return mse, nan_count

# Calculate MSE for each model
mse_plain, abstained_plain = calculate_mse(plain_residuals, perfect_answers)
mse_spatial, abstained_spatial = calculate_mse(spatial_residuals, perfect_answers)
mse_kg, abstained_kg = calculate_mse(kg_residuals, perfect_answers)


report_text = f"""
MSE for GPT-4: {mse_plain:.2e} [{len(plain_residuals) - abstained_plain}/{len(plain_residuals)} abstained]
MSE for SpatialRAG - Vector Similarity: {mse_spatial:.2e} [{len(spatial_residuals) - abstained_spatial}/{len(spatial_residuals)} abstained]
MSE for SpatialRAG - SPARQL: {mse_kg:.2e} [{len(kg_residuals) - abstained_kg}/{len(kg_residuals)} abstained]
"""


with open("mse_report_med.txt", "w") as file:
    file.write(report_text)

print(report_text) 