from dotenv import load_dotenv
from groq import Groq
import base64
from tqdm.auto import tqdm
from pathlib import Path
import os

image_dir = Path("data/v3/images")
caption_dir = Path("data/v3/captions_groq")
Trigger_Prefix = "LinkedIn Profile picture of Shl0k"

CAPTIONING_PROMPT = f"""You are writing a single image caption for a LoRA
fine-tuning dataset. The subject is a man whose trigger token is "Shl0k". 
Produce ONE caption following ALL rules below. 

HARD REQUIREMENTS:
- The caption MUST begin, word-for-word, with: "{Trigger_Prefix}". No other opening is acceptable.
- Output ONLY the caption itself. No preamble, no explaination, no quotes, no markdowns, no bullet points, no trailing notes.
- Length: 2 to 4 sentences, plain prose. 

DESCRIBE (only what is clearly visible):
- Physical features: hair color and type (e.g. short dark wavy hair, short black straight hair), glasses frame color and shape if worn, presence or absence of facial hair, skin tone described objectively (e.g. medium-brown skin) only if clearly visible.
- Clothing: exact colors and garment type (e.g. navy blue suit, white polo shirt, black blazer over white shirt, cream kurta, grey t-shirt).
Mention visible layers.
- Eyewear: whether glasses are worn; frame style/color if clearly visible.
- Pose and body language: e.g. arms raised, hands in pockets, leaning forward, standing straight, holding  microphone, seated.
- Facial Expression: smiling, serious, neutral, or confident — only if clearly visible.
- Gaze direction: looking at the camera, looking sideways, looking down.
- Lighting: stage lighting, natural daylight, indoor warm light, outdoor evening light, etc.
- Setting context in 3-5 words maximum (e.g. conference stage with screen, plain white wall, city balcony at night, office interior).
- Props clearly in hand (microphone, laptop, phone) only if visible.

STRICTLY FORBIDDEN:
- Do NOT use any of these phrases: "The image shows", "The image
is", "In this photo", "This photograph depicts", "Pictured here","We can see".
- Do NOT hedge: no "appears to be", "seems to", "looks like", "suggesting", "as if", "possibly", "perhaps".
- Do NOT describe mood, atmosphere, vibe, or aura: no words like "exciting", "professional aura", "dramatic atmosphere",
"energetic", "serene", "powerful presence".
- Do NOT guess age or ethnicity.
- Do NOT describe background elements that are not directly relevant to the subject.
- Do NOT mention anything you are not certain about. 
- Do NOT use bullet points or lists — plain prose only.

Now write the caption."""

caption_dir.mkdir(parents=True, exist_ok=True)
load_dotenv()
client = Groq(api_key = os.environ["GROQ_API_KEY"])
images = sorted(
    [p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}],
    key=lambda p: int(p.stem)
)
for img in tqdm(images, desc = "Captioning"):
    out = caption_dir/f"{img.stem}.txt"
    if out.exists():
        continue

    b64 = base64.b64encode(img.read_bytes()).decode()
    response = client.chat.completions.create(
        model = "meta-llama/llama-4-scout-17b-16e-instruct",
        messages = [{
            "role" : "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": CAPTIONING_PROMPT}
            ]
        }],
        temperature = 0.1,
        max_tokens = 512
    )

    if response:
        caption = response.choices[0].message.content.strip()
        out.write_text(caption, encoding = "utf-8")
        tqdm.write(f"{img.name}: {caption[:80]}...")
