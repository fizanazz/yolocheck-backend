"""
AI Health Assistant — powered by Google Gemini REST API.
Falls back to local responses when Gemini quota is exceeded.
"""
from __future__ import annotations
import logging
import httpx
from typing import Optional

from app.core.config import get_settings
from app.db.supabase_client import get_supabase
from app.schemas.chat import ChatMessage

logger = logging.getLogger("yolocheck.chat")

SYSTEM_PROMPT = (
    "You are YOLOCheck's AI Health Assistant — a friendly, empathetic, and knowledgeable "
    "skin-health educator. Your role is strictly educational.\n\n"
    "STRICT RULES — NEVER VIOLATE THESE:\n"
    "1. You MUST NOT diagnose any skin condition, disease, or cancer.\n"
    "2. You MUST NOT prescribe, recommend, or suggest any medication or treatment.\n"
    "3. You MUST NOT make definitive medical statements.\n"
    "4. You MUST always remind users that your output is AI-based screening, not medical advice.\n"
    "5. If the user asks for a diagnosis, politely decline and redirect to a dermatologist.\n\n"
    "WHAT YOU CAN DO:\n"
    "- Explain the ABCD rule of dermoscopy.\n"
    "- Explain what ABCD scores mean in plain language.\n"
    "- Provide general skin-care awareness tips.\n"
    "- Explain what a dermatologist does and why consulting one is important.\n"
    "- For HIGH-RISK scan results, strongly encourage seeing a dermatologist promptly.\n"
    "- Answer general educational questions about moles, skin health, and UV protection.\n"
    "- Explain the difference between benign and malignant moles.\n"
    "- Explain skin diseases like melanoma, eczema, psoriasis, carcinoma, rosacea.\n\n"
    "TONE: Warm, reassuring, and clear. Avoid medical jargon.\n"
    "Always end every response with:\n"
    "⚠️ Reminder: This is AI-generated educational information only — not a medical diagnosis. "
    "Please consult a qualified dermatologist for professional advice."
)

DISCLAIMER = (
    "⚠️ Reminder: This is AI-generated educational information only — not a medical diagnosis. "
    "Please consult a qualified dermatologist for professional advice."
)

GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)


# ── Local fallback answers ────────────────────────────────────────────────────

def _local_fallback(message: str) -> str:
    """Local answers when Gemini quota is exceeded."""
    q = message.lower()

    if ("benign" in q and "malignant" in q) or "difference" in q:
        return (
            "**Benign vs Malignant Moles:**\n\n"
            "• **Benign** moles are non-cancerous. They are usually uniform in color, "
            "have smooth borders, are symmetric, and stay the same over time. "
            "They pose no immediate health risk.\n\n"
            "• **Malignant** moles are cancerous or pre-cancerous. They may be asymmetric, "
            "have irregular borders, show multiple colors, or grow larger than 6mm. "
            "They require immediate medical attention.\n\n"
            "**Malignant is more dangerous** — it means the cells are abnormal and can "
            "spread to other parts of the body if untreated."
        )
    if "benign" in q:
        return (
            "**What is a Benign Mole?**\n\n"
            "A benign mole is a non-cancerous skin growth. Most people have 10–40 benign "
            "moles. They are typically:\n"
            "• Round or oval with smooth borders\n"
            "• One uniform color (tan, brown, or black)\n"
            "• Smaller than 6mm\n"
            "• Symmetric — both halves look the same\n"
            "• Stable — not changing over time\n\n"
            "Benign moles generally don't require treatment but should be monitored regularly."
        )
    if "malignant" in q:
        return (
            "**What is a Malignant Mole?**\n\n"
            "A malignant mole contains cancerous cells. The most serious type is melanoma. "
            "Warning signs include:\n"
            "• **Asymmetry** — one half doesn't match the other\n"
            "• **Border** — edges are irregular or ragged\n"
            "• **Color** — multiple shades of brown, black, red, or white\n"
            "• **Diameter** — larger than 6mm\n"
            "• **Evolution** — changing in size, shape, or color\n\n"
            "If you notice these signs, see a dermatologist immediately."
        )
    if "dangerous" in q or "more dangerous" in q:
        return (
            "**Which is More Dangerous?**\n\n"
            "**Malignant** is far more dangerous than benign.\n\n"
            "• Benign moles are harmless and don't spread.\n"
            "• Malignant moles (especially melanoma) can spread to lymph nodes and "
            "other organs if not treated early.\n\n"
            "Melanoma is the deadliest form of skin cancer but is highly treatable "
            "when caught early. This is why early detection tools like YOLOCheck are important."
        )
    if "turn" in q or "become" in q:
        return (
            "**Can a Benign Mole Turn Malignant?**\n\n"
            "Yes, although it is uncommon. Risk factors include:\n"
            "• Excessive UV/sun exposure\n"
            "• Family history of melanoma\n"
            "• Having many moles (50+)\n"
            "• Fair skin that burns easily\n"
            "• Atypical or dysplastic moles\n\n"
            "This is why regular self-examination and annual dermatologist visits are important."
        )
    if "melanoma" in q:
        return (
            "**What is Melanoma?**\n\n"
            "Melanoma is the most serious type of skin cancer. It develops in melanocytes "
            "(pigment-producing cells). Key facts:\n"
            "• It can appear as a new mole or develop within an existing one\n"
            "• It can spread to other organs if not caught early\n"
            "• It is highly treatable when detected early (5-year survival rate >98%)\n"
            "• Risk factors include UV exposure, fair skin, and family history\n\n"
            "Early detection through YOLOCheck and regular dermatologist visits "
            "is the best defense against melanoma."
        )
    if "basal" in q:
        return (
            "**What is Basal Cell Carcinoma?**\n\n"
            "Basal cell carcinoma (BCC) is the most common type of skin cancer. "
            "It grows slowly and rarely spreads. It appears as:\n"
            "• A pearly or waxy bump\n"
            "• A flat, flesh-colored lesion\n"
            "• A bleeding or scabbing sore that heals and returns\n\n"
            "BCC is caused mainly by long-term UV exposure and is highly treatable "
            "when caught early."
        )
    if "squamous" in q:
        return (
            "**What is Squamous Cell Carcinoma?**\n\n"
            "Squamous cell carcinoma (SCC) is the second most common skin cancer. "
            "It can spread if untreated. Signs include:\n"
            "• A firm, red nodule\n"
            "• A flat lesion with a scaly, crusted surface\n"
            "• A new sore on an old scar\n\n"
            "SCC is caused by UV exposure and is treatable when detected early."
        )
    if "eczema" in q:
        return (
            "**What is Eczema?**\n\n"
            "Eczema (atopic dermatitis) is a chronic inflammatory skin condition causing:\n"
            "• Dry, itchy, inflamed skin\n"
            "• Red to brownish-gray patches\n"
            "• Small, raised bumps that may weep fluid\n\n"
            "It is not contagious or cancerous. It is managed with moisturizers "
            "and avoiding triggers."
        )
    if "psoriasis" in q:
        return (
            "**What is Psoriasis?**\n\n"
            "Psoriasis is a chronic autoimmune condition causing rapid skin cell buildup. "
            "Symptoms include:\n"
            "• Red patches covered with thick, silvery scales\n"
            "• Dry, cracked skin that may bleed\n"
            "• Itching, burning, or soreness\n\n"
            "It is not contagious. Treatment includes topical creams and light therapy."
        )
    if "rosacea" in q:
        return (
            "**What is Rosacea?**\n\n"
            "Rosacea is a chronic skin condition causing redness mainly on the face. "
            "Symptoms include:\n"
            "• Facial redness and flushing\n"
            "• Visible blood vessels\n"
            "• Swollen red bumps\n\n"
            "Triggers include sun exposure, hot drinks, and stress. "
            "It is manageable but not curable."
        )
    if "dermatitis" in q:
        return (
            "**What is Dermatitis?**\n\n"
            "Dermatitis is inflammation of the skin. Types include:\n"
            "• **Contact dermatitis** — reaction to something touching the skin\n"
            "• **Atopic dermatitis (eczema)** — chronic itchy inflammation\n"
            "• **Seborrheic dermatitis** — flaky patches on oily areas\n\n"
            "Treatment depends on the type but usually involves avoiding triggers "
            "and using topical treatments."
        )
    if "seborrheic" in q:
        return (
            "**What is Seborrheic Keratosis?**\n\n"
            "Seborrheic keratosis is a common non-cancerous skin growth. Features:\n"
            "• Waxy, scaly, slightly raised appearance\n"
            "• Light tan to black color\n"
            "• Looks 'stuck on' the skin\n"
            "• Usually appears after age 50\n\n"
            "It is completely benign and requires no treatment unless irritated."
        )
    if "abcd" in q:
        return (
            "**The ABCD Rule of Dermoscopy:**\n\n"
            "• **A — Asymmetry**: One half of the mole doesn't match the other\n"
            "• **B — Border**: Edges are irregular, ragged, notched, or blurred\n"
            "• **C — Color**: Variation in color (shades of brown, black, red, white)\n"
            "• **D — Diameter**: Larger than 6mm (size of a pencil eraser)\n\n"
            "YOLOCheck scores each 0–3 for a Total Dermoscopy Score:\n"
            "• 0–4: Low Risk\n"
            "• 5–8: Moderate Risk\n"
            "• 9–12: High Risk"
        )
    if "asymmetry" in q:
        return (
            "**A — Asymmetry in the ABCD Rule:**\n\n"
            "Asymmetry means one half of the mole does not mirror the other half. "
            "A normal mole is symmetric — if you drew a line through the middle, "
            "both halves would match.\n\n"
            "YOLOCheck scores asymmetry 0–3:\n"
            "• 0–1: Symmetric (reassuring)\n"
            "• 2–3: Significant asymmetry (warrants evaluation)"
        )
    if "border" in q:
        return (
            "**B — Border in the ABCD Rule:**\n\n"
            "Border refers to the edges of the mole. Benign moles have smooth, "
            "well-defined borders. Concerning moles have:\n"
            "• Irregular or ragged edges\n"
            "• Notched or scalloped borders\n"
            "• Poorly defined edges that fade into surrounding skin\n\n"
            "YOLOCheck scores border irregularity 0–3."
        )
    if "color" in q:
        return (
            "**C — Color in the ABCD Rule:**\n\n"
            "Color variation within a mole is a warning sign. Benign moles are "
            "usually one uniform shade. Concerning moles may show:\n"
            "• Multiple shades of brown or black\n"
            "• Red, white, or blue areas\n"
            "• Uneven pigmentation\n\n"
            "YOLOCheck scores color variation 0–3."
        )
    if "diameter" in q:
        return (
            "**D — Diameter in the ABCD Rule:**\n\n"
            "The clinical threshold for concerning diameter is 6mm — about the size "
            "of a pencil eraser. Moles larger than this warrant evaluation.\n\n"
            "However, some melanomas can be smaller than 6mm, so size alone is not "
            "the only factor. YOLOCheck scores diameter 0–3 based on relative size."
        )
    if "high risk" in q:
        return (
            "**High Risk Classification:**\n\n"
            "A High Risk result means YOLOCheck detected significant ABCD features. "
            "This does not mean you have cancer, but it strongly suggests you should "
            "consult a board-certified dermatologist within 1 week for professional evaluation."
        )
    if "low risk" in q:
        return (
            "**Low Risk Classification:**\n\n"
            "A Low Risk result means the mole's ABCD features are within normal ranges. "
            "Continue monthly self-examinations, use SPF 50+ sunscreen daily, "
            "and schedule annual professional skin checks."
        )
    if "moderate risk" in q:
        return (
            "**Moderate Risk Classification:**\n\n"
            "Some features of potential concern were identified. Schedule a dermatology "
            "appointment within 2–4 weeks and track any changes in size, shape, or color."
        )
    if "confidence" in q:
        return (
            "**Confidence Score:**\n\n"
            "The confidence score shows how certain YOLOv11 is about its detection. "
            "90% confidence means the model is 90% sure the detected region matches "
            "the patterns it was trained on. Scores above 80% are generally reliable."
        )
    if "total" in q and "score" in q:
        return (
            "**Total Dermoscopy Score (TDS):**\n\n"
            "The TDS is the sum of all four ABCD scores (each 0–3), giving a total of 0–12:\n"
            "• **0–4**: Low Risk — features within normal range\n"
            "• **5–8**: Moderate Risk — some concerning features\n"
            "• **9–12**: High Risk — significant concerning features\n\n"
            "Your score reflects the combined assessment of asymmetry, border, color, and diameter."
        )
    if "prevent" in q or "prevention" in q:
        return (
            "**Preventing Skin Cancer:**\n\n"
            "• Apply SPF 50+ broad-spectrum sunscreen daily\n"
            "• Avoid sun between 10am–4pm\n"
            "• Wear protective clothing and wide-brimmed hats\n"
            "• Never use tanning beds\n"
            "• Perform monthly self-skin examinations\n"
            "• Get annual professional skin checks\n"
            "• Check moles using the ABCD rule regularly"
        )
    if "dermatologist" in q or "doctor" in q or "when should" in q:
        return (
            "**When to See a Dermatologist:**\n\n"
            "• **Immediately** for High Risk results\n"
            "• **Within 2–4 weeks** for Moderate Risk results\n"
            "• **Annually** for routine skin checks\n"
            "• **Any time** a mole changes, bleeds, itches, or looks unusual\n\n"
            "Early detection is the most important factor in successful skin cancer treatment."
        )
    if "spf" in q or "sunscreen" in q:
        return (
            "**Sunscreen Recommendations:**\n\n"
            "• Use **SPF 50+** broad-spectrum sunscreen\n"
            "• Apply 15–30 minutes before sun exposure\n"
            "• Reapply every 2 hours and after swimming\n"
            "• Use on all exposed skin year-round\n\n"
            "Sunscreen is the single most effective tool for preventing skin cancer."
        )
    if "uv" in q or "sun" in q:
        return (
            "**How UV Radiation Affects Skin:**\n\n"
            "UV radiation damages DNA in skin cells, which can cause mutations leading "
            "to skin cancer. There are two types:\n"
            "• **UVA** — penetrates deeply, causes aging and DNA damage\n"
            "• **UVB** — causes sunburn and is the main cause of skin cancer\n\n"
            "Both types contribute to melanoma risk. Use broad-spectrum SPF 50+ to "
            "block both UVA and UVB rays."
        )
    if "food" in q or "diet" in q:
        return (
            "**Foods That Help Skin Health:**\n\n"
            "• **Tomatoes** — lycopene protects against UV damage\n"
            "• **Green tea** — antioxidants reduce skin inflammation\n"
            "• **Fatty fish** — omega-3s reduce inflammation\n"
            "• **Carrots & sweet potatoes** — beta-carotene supports skin repair\n"
            "• **Dark chocolate** — flavonoids improve skin hydration\n"
            "• **Broccoli** — sulforaphane has anti-cancer properties\n\n"
            "A balanced diet supports overall skin health but does not replace sunscreen."
        )
    if "check" in q or "self" in q or "examine" in q:
        return (
            "**How to Check Your Moles:**\n\n"
            "Perform a self-examination monthly:\n"
            "1. Use a full-length mirror in good lighting\n"
            "2. Check your entire body including scalp, between toes, and under nails\n"
            "3. Use a hand mirror for hard-to-see areas\n"
            "4. Apply the ABCD rule to each mole\n"
            "5. Take photos to track changes over time\n\n"
            "See a dermatologist immediately if any mole changes suddenly."
        )
    if "yolo" in q or "how does" in q or "detect" in q:
        return (
            "**How YOLOv11 Detects Moles:**\n\n"
            "YOLOv11 (You Only Look Once v11) is a real-time object detection neural network. "
            "YOLOCheck's model was trained on thousands of dermoscopy images to:\n"
            "• Locate moles with precise bounding boxes\n"
            "• Classify them as benign or malignant\n"
            "• Provide a confidence score\n\n"
            "The ABCD analysis is then performed on the detected region for additional "
            "risk assessment."
        )
    if "hello" in q or "hi" in q or "hey" in q:
        return (
            "Hello! I'm the YOLOCheck AI Health Assistant. I can help you understand:\n"
            "• Your scan results (benign/malignant, risk level, ABCD scores)\n"
            "• The difference between benign and malignant moles\n"
            "• Skin diseases like melanoma, eczema, psoriasis\n"
            "• When to see a dermatologist\n"
            "• How to prevent skin cancer\n\n"
            "What would you like to know?"
        )

    return (
        "I can help you understand your YOLOCheck scan results, the ABCD dermoscopy rule, "
        "the difference between benign and malignant moles, skin diseases like melanoma, "
        "eczema, and psoriasis, and when to consult a dermatologist.\n\n"
        "Please ask me a specific question about any of these topics!"
    )


# ── Scan context ──────────────────────────────────────────────────────────────

def _fetch_scan_context(scan_id: str) -> str:
    try:
        supabase = get_supabase()
        scan = supabase.table("scans").select("*").eq("id", scan_id).single().execute()
        if not scan.data:
            return ""
        s = scan.data
        dets = (
            supabase.table("detections")
            .select("mole_id, risk_level, abcd_asymmetry, abcd_border, abcd_color, abcd_diameter, abcd_total")
            .eq("scan_id", scan_id)
            .execute()
        )
        lines = [
            f"The user's scan detected {s['total_moles_detected']} mole(s).",
            f"Overall highest risk level: {s['highest_risk']}.",
        ]
        for d in (dets.data or []):
            lines.append(
                f"  • {d['mole_id']}: Risk={d['risk_level']}, "
                f"ABCD total={d['abcd_total']} "
                f"(A={d['abcd_asymmetry']}, B={d['abcd_border']}, "
                f"C={d['abcd_color']}, D={d['abcd_diameter']})"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Could not load scan context: %s", exc)
        return ""


# ── Main chat function ────────────────────────────────────────────────────────

def chat(
    message: str,
    scan_id: Optional[str] = None,
    user_id: Optional[str] = None,
    history: Optional[list[ChatMessage]] = None,
) -> str:
    settings = get_settings()

    context = ""
    if scan_id:
        context = _fetch_scan_context(scan_id)

    history_text = ""
    if history:
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role_label}: {msg.content}\n"

    parts = [SYSTEM_PROMPT]
    if context:
        parts.append(f"[Scan Context]\n{context}")
    if history_text:
        parts.append(f"[Conversation History]\n{history_text.strip()}")
    parts.append(f"[User Question]\n{message}")

    full_prompt = "\n\n".join(parts)

    url = GEMINI_REST_URL.format(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}],
                "role": "user",
            }
        ],
        "generationConfig": {
            "temperature":     0.7,
            "maxOutputTokens": 1024,
        },
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data  = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            if "not a medical diagnosis" not in reply.lower():
                reply = f"{reply}\n\n{DISCLAIMER}"

            return reply

    except Exception as exc:
        # Gemini unavailable — use local fallback
        logger.warning("Gemini unavailable (%s) — using local fallback.", exc)
        return _local_fallback(message) + "\n\n" + DISCLAIMER