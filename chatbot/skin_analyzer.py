import base64
from openai import OpenAI

client = OpenAI()


def analyze_skin(image_path):

    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()

    response = client.responses.create(

        model="gpt-4.1",

        input=[
            {
                "role": "user",
                "content": [

                    {
                        "type": "input_text",
                        "text": """
You are a dermatology AI.

Analyze this face image and determine:

1. Skin type:
- oily
- dry
- combination
- normal
- sensitive

2. Skin concerns:
- acne
- redness
- enlarged pores
- dryness
- pigmentation

Return EXACTLY this format:

Skin Type: <type>

Observations:
- ...
- ...
"""
                    },

                    {
                        "type": "input_image",
                        "image_base64": img
                    }
                ]
            }
        ]
    )

    return response.output_text