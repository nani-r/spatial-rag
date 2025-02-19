import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from matplotlib.ticker import MaxNLocator

# Initialize geolocator
geolocator = Nominatim(user_agent="geo_distance_lookup")

CACHE_FILE = "cached_distances.json"

def load_cache():
    """Loads cached computed distances if available."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Saves computed distances to cache file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def get_coordinates(city_name):
    """Returns latitude and longitude of a city using geopy."""
    try:
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
    except:
        pass
    return None 

def extract_cities_from_question(question):
    """Extracts cities from the question using regex and returns the third city."""
    city_matches = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", question)
    return city_matches[2] if len(city_matches) >= 3 else None  

def extract_geodesic_distance(question, answer, cache):
    """Computes or retrieves cached geodesic distance, handling null answers."""
    if answer is None: 
        return None
    
    third_city = extract_cities_from_question(question)
    if third_city:
        cache_key = f"{third_city} -> {answer}" 
        if cache_key in cache:
            return cache[cache_key]  
        
        third_city_coords = get_coordinates(third_city)
        answer_coords = get_coordinates(answer)
        if third_city_coords and answer_coords:
            distance = geodesic(third_city_coords, answer_coords).km  
            cache[cache_key] = distance 
            return distance
    return None  

def load_json(filename):
    """Loads a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)

def extract_answers(data, cache):
    """Extracts computed geodesic distances for Hard questions."""
    distances = []
    
  
    if isinstance(data, dict) and "answers" in data:
        data = data["answers"] 
    
    for item in data:
        if item and "hard" in item:  
            question = item["hard"]["question"]
            answer = item["hard"].get("answer")  
            distance = extract_geodesic_distance(question, answer, cache)
            distances.append(distance)
    
    return distances

# Load JSON data
plain_llm = load_json("llm_answers.json")
spatial_rag = load_json("spatialrag_answers.json")
kg_answers = load_json("kg_answers.json")
city_questions = load_json("city_questions.json")

# Load or initialize cache
cache = load_cache()

# Extract computed distances for Hard difficulty questions
plain_answers = extract_answers(plain_llm, cache)
spatial_answers = extract_answers(spatial_rag, cache)
kg_answers = extract_answers(kg_answers, cache)  
perfect_answers = extract_answers(city_questions, cache)  
print(perfect_answers)
# Save updated cache
save_cache(cache)

# Compute residuals and ensure they fall in the correct bins
def process_residuals(predicted, ground_truth):
    """Processes residuals, converting None to NaN and applying binning."""
    processed = []
    for a, p in zip(predicted, ground_truth):
        if a is None or p is None or np.isnan(a) or np.isnan(p) or abs(a - p) > 700:
            processed.append(np.nan)  
        else:
            processed.append(abs(a - p))
    return processed


def calculate_mse(residuals, perfect_answers):
    # Filter out the residuals that are None
    valid_residuals = [(r, p) for r, p in zip(residuals, perfect_answers) if r is not None and p is not None]
    
    if len(valid_residuals) == 0:
        return 0, len(residuals)  
    

    residuals_valid, perfect_answers_valid = zip(*valid_residuals)
    mse = np.mean(np.square(np.array(residuals_valid)))
    abstained_count = len(residuals) - len(valid_residuals)
    
    return mse, abstained_count

plain_residuals = [341.2113185043429, None, 108.2416155150811, 39.9419496567549, 17382.76367570304, None, 86.07700931020088, 484.89820081212207, 576.719617296354, 172.19703013130857, 435.634271803101, 12575.080444701784, 740.6058124186432, 0.0, 311.17221991864153, 16552.732166984402, 1231.9062299739344, 1180.961996221275, 505.0930984218535, 0.0]
spatial_residuals = [1223.819099344773, 0.0, 239.9851148004617, None, 14245.919341531615, 186.7164780315952, 789.2624730824605, 357.85076520606503, 1816.1982224107378, 15449.269274524562, 16766.507354772213, 197.33801477389443, 118.22863462556438, 14703.395127360915, 311.17221991864153, 12948.329548997324, 1231.9062299739344, 15999.43945623923, 492.92715599559415, 14620.15536062053]
kg_residuals = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
perfect_answers = [366.06617050505645, 0.0, 832.0635211442923, 690.3959126981213, 17382.76367570304, 186.7164780315952, 854.9736636081489, 554.6582572561622, 14953.59811758999, 16505.99457014566, 16766.507354772213, 2537.837682849139, 1868.3175779434916, 15336.389480761964, 311.17221991864153, 16552.732166984402, 1911.5478219254842, 328.5489124088459, 3594.4291003018125, 1696.0240789735265]

mse_plain, abstained_plain = calculate_mse(plain_residuals, perfect_answers)
mse_spatial, abstained_spatial = calculate_mse(spatial_residuals, perfect_answers)
mse_kg, abstained_kg = calculate_mse(kg_residuals, perfect_answers)

report_text = f"""
MSE for GPT-4: {mse_plain:.2e} [{len(plain_residuals) - abstained_plain}/{len(plain_residuals)} abstained]
MSE for SpatialRAG - Vector Similarity: {mse_spatial:.2e} [{len(spatial_residuals) - abstained_spatial}/{len(spatial_residuals)} abstained]
MSE for SpatialRAG - SPARQL: {mse_kg:.2e} [{len(kg_residuals) - abstained_kg}/{len(kg_residuals)} abstained]
"""

# Write the report to a text file
with open("mse_report_hard.txt", "w") as file:
    file.write(report_text)

print(report_text) 


# Calculate residuals for histogram
plain_residuals = process_residuals(plain_answers, perfect_answers)
spatial_residuals = process_residuals(spatial_answers, perfect_answers)
kg_residuals = process_residuals(kg_answers, perfect_answers)


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
