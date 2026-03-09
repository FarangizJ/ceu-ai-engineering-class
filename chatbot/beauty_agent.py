from pathlib import Path
import os
 
import chromadb
import requests
from agents import Agent, FunctionTool, function_tool
import dotenv
 
dotenv.load_dotenv()
 
MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"
 
 
def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )
 
 
# -----------------------------------------
# Build / load beauty ingredient RAG DB
# -----------------------------------------
 
DATA_PATH = Path(__file__).parent.parent / "data" / "beauty_ingredients.txt"
CHROMA_PATH = Path(__file__).parent.parent / "chroma"
COLLECTION_NAME = "beauty_ingredients_db"
 
 
def parse_ingredient_lines(path: Path) -> list[tuple[str, str]]:
    """Parse `ingredient: description` rows from the source text file."""
    items: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, description = line.split(":", 1)
            items.append((name.strip(), description.strip()))
    return items
 
 
def build_or_load_beauty_collection():
    """Create and seed the Chroma collection if it does not exist yet."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
 
    if collection.count() == 0 and DATA_PATH.exists():
        items = parse_ingredient_lines(DATA_PATH)
        if items:
            documents = [
                f"Ingredient: {name}\nBenefits: {desc}"
                for name, desc in items
            ]
            metadatas = [
                {
                    "ingredient": name.lower(),
                    "benefits": desc,
                }
                for name, desc in items
            ]
            ids = [f"ingredient_{i}" for i in range(len(items))]
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
 
    return collection
 
 
beauty_db = build_or_load_beauty_collection()
 
 
# -----------------------------------------
# Ingredient lookup tool
# -----------------------------------------
 
@function_tool
def ingredient_lookup(query: str, max_results: int = 3) -> str:
    """Semantic lookup for skincare ingredient uses and benefits."""
    try:
        results = beauty_db.query(query_texts=[query], n_results=max_results)
    except Exception as e:
        return f"Ingredient search error: {str(e)}"
 
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
 
    if not docs:
        return f"No ingredient information found for: {query}"
 
    lines = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        name = meta.get("ingredient", "unknown ingredient").title()
        lines.append(f"{i + 1}. {name}: {doc}")
 
    return "Skincare ingredient knowledge base results:\n" + "\n".join(lines)
 
 
# -----------------------------------------
# Web search for cosmetics
# -----------------------------------------
 
@function_tool
def search_cosmetics_shops(product: str, location: str = "Vienna") -> str:
    """
    Search online where a cosmetic product can be purchased in a given location.
    """
 
    api_key = os.environ.get("EXA_API_KEY")
 
    if not api_key:
        return f"Online search unavailable. Try checking local pharmacies or cosmetic stores in {location}."
 
    query = f"where to buy {product} skincare in {location}"
 
    url = "https://api.exa.ai/search"
 
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
 
    payload = {
        "query": query,
        "numResults": 5
    }
 
    try:
        response = requests.post(url, json=payload, headers=headers)
 
        if response.status_code != 200:
            return "Search service returned an error."
 
        data = response.json()
 
        results = []
 
        for r in data.get("results", []):
            title = r.get("title", "Unknown store")
            url = r.get("url", "")
 
            results.append(f"{title}\n{url}")
 
        if results:
            return f"Online places to buy {product} in {location}:\n\n" + "\n\n".join(results)
 
    except Exception as e:
        return f"Search error: {str(e)}"
 
    return f"I couldn't find online shops for {product} in {location}, but try pharmacies or beauty retailers."
 
 
# -----------------------------------------
# Main AI Beauty Advisor
# -----------------------------------------
 
beauty_agent = Agent(
    name="Beauty Advisor",
 
    instructions="""
You are an AI skincare advisor.
 
Your role is to help users understand their skin condition, recommend routines, and suggest useful skincare ingredients.
 
------------------------------------------------
GUARDRAILS (IMPORTANT)
 
You must ONLY answer questions related to:
 
• skincare
• skin problems
• cosmetic ingredients
• skincare routines
• acne, pores, redness, dryness
• cosmetic products
• skin analysis results
 
If a user asks about topics unrelated to skincare (politics, math, coding, etc.), respond:
 
"I'm a skincare assistant and can only help with skincare-related questions."
 
------------------------------------------------
MEDICAL SAFETY
 
You are NOT a medical professional.
 
You must NOT:
 
• diagnose diseases
• prescribe medications
• recommend prescription drugs
• recommend medical procedures
 
If the user asks about serious or persistent skin conditions, respond:
 
"For persistent or severe skin conditions, it is best to consult a dermatologist."
 
------------------------------------------------
SELFIE / SKIN ANALYSIS
 
Sometimes the system provides a Skin Analysis Report generated by a computer vision system that analyzed the user's face.
 
You MUST assume the analysis is correct.
 
Do NOT say you cannot analyze images.
 
Instead, interpret the analysis and provide advice.
 
------------------------------------------------
WHEN A SKIN ANALYSIS IS PROVIDED
 
You must:
 
1. Identify the skin type
2. Explain the main skin conditions
3. Provide a skincare routine tailored to that skin type
4. Recommend useful ingredients
 
Use this structure:
 
Skin Type:
<type>
 
Explanation:
<short explanation>
 
Morning Routine:
- step
- step
- step
 
Evening Routine:
- step
- step
- step
 
Recommended Ingredients:
- ingredient
- ingredient
 
------------------------------------------------
INGREDIENT QUESTIONS
 
If the user asks about a skincare ingredient,
use the ingredient_lookup tool.
 
For ingredient questions, run ingredient_lookup first and use the retrieved information in your answer.
 
------------------------------------------------
BUYING PRODUCTS
 
If the user asks where to buy a product,
use the search_cosmetics_shops tool.
 
------------------------------------------------
TONE
 
Be helpful, friendly, and professional like a skincare consultant.
""",
 
    model=MODEL,
 
    tools=[
        bedrock_tool(ingredient_lookup.__dict__),
        bedrock_tool(search_cosmetics_shops.__dict__),
    ]
)
