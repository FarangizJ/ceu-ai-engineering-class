from agents import Agent, function_tool
import requests
import os


# -----------------------------------------
# Ingredient knowledge base
# -----------------------------------------

beauty_db = {

    "niacinamide": "Vitamin B3 that reduces oil production, improves enlarged pores, strengthens the skin barrier, and helps with acne and redness.",

    "retinol": "A vitamin A derivative that accelerates skin cell turnover, reduces wrinkles, improves acne, and stimulates collagen production.",

    "salicylic acid": "A beta hydroxy acid (BHA) that penetrates pores, dissolves excess oil, unclogs pores, and treats acne and blackheads.",

    "hyaluronic acid": "A powerful humectant that attracts and retains moisture in the skin, improving hydration and reducing dryness.",

    "azelaic acid": "Helps treat acne, reduce redness, calm inflammation, and improve pigmentation and rosacea.",

    "ceramides": "Lipids naturally found in the skin barrier that help retain moisture and protect against irritation and dryness.",

    "vitamin c": "A powerful antioxidant that brightens skin, boosts collagen production, and reduces dark spots and pigmentation.",

    "benzoyl peroxide": "Kills acne-causing bacteria, reduces inflammation, and helps clear pimples and cystic acne.",

    "glycolic acid": "An alpha hydroxy acid (AHA) that exfoliates the skin surface, improves texture, and helps reduce pigmentation and fine lines.",

    "lactic acid": "A gentle AHA that exfoliates while also hydrating the skin, improving brightness and smoothness.",

    "zinc": "Helps regulate oil production, reduce inflammation, and support acne-prone skin.",

    "panthenol": "Also known as provitamin B5, it hydrates, soothes irritation, and helps repair the skin barrier.",

    "centella asiatica": "A soothing plant extract that reduces redness, supports healing, and strengthens the skin barrier.",

    "tea tree oil": "A natural antibacterial ingredient that helps reduce acne and inflammation.",

    "squalane": "A lightweight moisturizing oil that hydrates the skin without clogging pores.",

    "peptides": "Short chains of amino acids that support collagen production and help improve skin firmness and elasticity.",

    "alpha arbutin": "A skin-brightening ingredient that reduces dark spots and hyperpigmentation.",

    "kojic acid": "Helps lighten pigmentation and dark spots by inhibiting melanin production.",

    "tranexamic acid": "Reduces hyperpigmentation and melasma by interfering with melanin production pathways.",

    "green tea extract": "An antioxidant-rich ingredient that reduces inflammation, redness, and excess oil production.",

    "allantoin": "A soothing ingredient that promotes skin healing and reduces irritation.",

    "urea": "Helps hydrate dry skin and gently exfoliate dead skin cells."
}


@function_tool
def ingredient_lookup(ingredient: str) -> str:
    """Look up skincare ingredient benefits."""

    ingredient = ingredient.lower()

    if ingredient in beauty_db:
        return beauty_db[ingredient]

    return "Ingredient not found in the skincare database."


# -----------------------------------------
# Find stores in Vienna
# -----------------------------------------

@function_tool
def find_beauty_store(product: str, location: str = "Vienna") -> str:
    """
    Search where a cosmetic product can be bought in a specific location.
    """

    query = f"where to buy {product} cosmetics in {location}"

    api_key = os.environ.get("EXA_API_KEY")

    if not api_key:
        return f"Search unavailable. But common places to buy cosmetics in {location} include pharmacies and beauty stores."

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
        data = response.json()

        results = []

        for r in data.get("results", []):
            results.append(f"{r['title']} — {r['url']}")

        if results:
            return f"Places to buy {product} in {location}:\n\n" + "\n".join(results)

    except Exception:
        pass

    return f"I couldn't find online results, but try pharmacies or cosmetic stores in {location}."

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

Your role is to help users understand their skin condition, recommend routines, and suggest useful ingredients or places to buy skincare products.

GENERAL RULES
- Be clear and concise.
- Answer only what the user asked.
- Do not show reasoning or internal thinking.
- Provide practical and safe skincare advice.
- Do not invent medical diagnoses.

------------------------------------------------

SELFIE / SKIN ANALYSIS

Sometimes the system provides a Skin Analysis Report generated by a computer vision system that analyzed the user's face.

You MUST assume the analysis is correct.

Do NOT say you cannot analyze images.

Instead, interpret the analysis and provide advice.

Example input:

Skin Analysis Report:
Skin Type: oily
Acne spots detected: 12
Oil level: high
Redness: moderate
Pore visibility: large

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
<short explanation of the skin condition>

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

Example:
User: What does niacinamide do?

------------------------------------------------

BUYING PRODUCTS

If the user asks where to buy a product:

1. Identify the product
2. Identify the user's location from the conversation if available
3. Use the store search tools

If no location is known, assume Vienna.

------------------------------------------------

LOCATION AWARENESS

The user may mention where they live.

Example:
"I live in Vienna"
"I am in Uzbekistan"

Remember the location from the conversation and use it when suggesting stores.

------------------------------------------------

TONE

Be helpful, friendly, and professional like a skincare consultant.
""",

    model="litellm/bedrock/eu.amazon.nova-lite-v1:0",

    tools=[
        ingredient_lookup,
        find_beauty_store,
        search_cosmetics_shops
    ]
)