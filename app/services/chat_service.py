"""
AI Health Assistant — powered by Google Gemini REST API.
Falls back to local responses when Gemini quota is exceeded.
Handles emotional, symptom, and human worried questions.
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
    "You are YOLOCheck's AI Health Assistant — a warm, empathetic, and knowledgeable "
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
    "- Explain skin diseases like melanoma, eczema, psoriasis, carcinoma, rosacea.\n"
    "- Provide emotional support and reassurance to worried users.\n"
    "- Answer questions about symptoms like itching, bleeding, changing moles.\n"
    "- Help users understand what to expect at a dermatologist appointment.\n"
    "- Answer questions about anxiety, fear, and worry about skin cancer.\n"
    "- Answer ANY question related to skin health, moles, or skin diseases.\n"
    "- If the user uploads a report, summarize and explain it clearly.\n\n"
    "REPORT HANDLING:\n"
    "- If the message contains [YOLOCHECK REPORT CONTENT], read it carefully.\n"
    "- Summarize: classification, risk level, confidence, ABCD scores, TDS.\n"
    "- Explain what the scores mean in simple language.\n"
    "- Give appropriate next steps based on the risk level.\n\n"
    "EMOTIONAL SUPPORT:\n"
    "- Be warm, calm, and reassuring when users express fear or anxiety.\n"
    "- Acknowledge their feelings before providing information.\n"
    "- Never dismiss concerns — treat every question with care and respect.\n\n"
    "TONE: Warm, reassuring, human, and clear. Avoid medical jargon.\n"
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


def _call_gemini(message: str, system: str, max_tokens: int = 1024) -> Optional[str]:
    """Call Gemini API with a given prompt. Returns None if unavailable."""
    settings = get_settings()
    url = GEMINI_REST_URL.format(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )
    payload = {
        "contents": [{"parts": [{"text": f"{system}\n\n{message}"}], "role": "user"}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": max_tokens},
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data  = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if "not a medical diagnosis" not in reply.lower():
                reply = f"{reply}\n\n{DISCLAIMER}"
            return reply
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None


def _extract_report_data(message: str) -> dict:
    """Extract key data from uploaded report text."""
    data      = {}
    msg_lower = message.lower()

    if "benign"    in msg_lower: data["classification"] = "BENIGN"
    elif "malignant" in msg_lower: data["classification"] = "MALIGNANT"

    if "high risk"     in msg_lower: data["risk"] = "High"
    elif "moderate risk" in msg_lower: data["risk"] = "Moderate"
    elif "low risk"      in msg_lower: data["risk"] = "Low"

    import re
    conf = re.search(r"confidence[:\s]+(\d+)%", msg_lower)
    if conf: data["confidence"] = conf.group(1)

    tds = re.search(r"total[:\s]+(\d+\.?\d*)\s*/\s*12", msg_lower)
    if tds: data["tds"] = tds.group(1)

    return data


def _local_fallback(message: str) -> str:
    """
    Local answers when Gemini quota is exceeded.
    Covers predefined topics first, then tries Gemini with a simpler prompt,
    then falls back to a helpful generic response.
    """
    q = message.lower()

    # ── Report summarization ───────────────────────────────────────────────────
    if "[yolocheck report content]" in q or any(w in q for w in [
        "summarize", "summary", "summarise",
        "what does my report say", "explain my report",
        "read my report", "what is in my report"
    ]):
        report_data    = _extract_report_data(message)
        classification = report_data.get("classification", "Unknown")
        risk           = report_data.get("risk", "Unknown")
        confidence     = report_data.get("confidence", "Unknown")
        tds            = report_data.get("tds", "Unknown")
        next_steps = {
            "High":     "See a dermatologist **within 1 week**.",
            "Moderate": "Schedule a dermatologist appointment **within 2–4 weeks**.",
            "Low":      "Continue monthly self-checks and annual professional skin checks.",
        }.get(risk, "Consult a dermatologist for professional evaluation.")
        return (
            f"**Your YOLOCheck Report Summary:**\n\n"
            f"• **Classification:** {classification}\n"
            f"• **Risk Level:** {risk} Risk\n"
            f"• **AI Confidence:** {confidence}%\n"
            f"• **Total Dermoscopy Score:** {tds} / 12\n\n"
            f"**What this means:**\n"
            f"Your mole was classified as **{classification}** with **{risk} Risk**.\n\n"
            f"**Next Steps:** {next_steps}\n\n"
            f"Remember: This is an AI screening result — only a qualified dermatologist "
            f"can give you a definitive diagnosis."
        )

    if any(w in q for w in ["risk level", "what is my risk", "my risk"]):
        return (
            "**Risk Levels Explained:**\n\n"
            "• **Low Risk** — Features within normal range. Continue routine monitoring.\n"
            "• **Moderate Risk** — Some concerning features. See a dermatologist within 2–4 weeks.\n"
            "• **High Risk** — Significant features detected. See a dermatologist within 1 week.\n\n"
            "Check your result screen for your specific risk level."
        )

    if any(w in q for w in ["what should i do next", "next steps",
                              "based on my results", "what to do"]):
        return (
            "**Your Next Steps:**\n\n"
            "• **High Risk** → Dermatologist within 1 week\n"
            "• **Moderate Risk** → Appointment within 2–4 weeks\n"
            "• **Low Risk** → Monthly self-checks + annual professional check\n\n"
            "Also: take photos of the mole to track changes and use SPF 50+ daily."
        )

    if any(w in q for w in ["explain my abcd", "what are my scores", "abcd scores from",
                              "abcd score", "my abcd"]):
        return (
            "**Your ABCD Scores (each rated 0–3):**\n\n"
            "• **A (Asymmetry)** — Is one half different from the other?\n"
            "• **B (Border)** — Are the edges irregular or ragged?\n"
            "• **C (Color)** — Are there multiple colors present?\n"
            "• **D (Diameter)** — Is it larger than 6mm?\n\n"
            "**Total Dermoscopy Score:** 0–4 Low · 5–8 Moderate · 9–12 High Risk"
        )

    if any(w in q for w in ["classification", "what does classification mean",
                              "benign or malignant result"]):
        return (
            "**Classification:**\n\n"
            "• **BENIGN** — AI detected characteristics of a harmless mole\n"
            "• **MALIGNANT** — AI detected potentially concerning characteristics\n\n"
            "This is NOT a diagnosis — only a dermatologist can confirm through "
            "proper clinical examination."
        )

    # ── Emotional questions ────────────────────────────────────────────────────
    if any(w in q for w in ["scared", "afraid", "terrified", "fear",
                              "worried", "worry", "anxious", "anxiety", "panic", "nervous"]):
        return (
            "I completely understand — it's completely normal to feel scared. 💙\n\n"
            "Here's what's important:\n"
            "• **Most moles are completely harmless** — vast majority are benign\n"
            "• Even concerning results are almost always treatable when caught early\n"
            "• YOLOCheck is a screening tool — it flags things for review, not a diagnosis\n\n"
            "The bravest thing you can do is **book a dermatologist appointment**. "
            "You are not alone. 💙"
        )

    if any(w in q for w in ["cancer", "do i have cancer", "is it cancer"]):
        return (
            "I understand why you're worried. I cannot tell you whether you have cancer — "
            "only a qualified dermatologist can determine that.\n\n"
            "What I can tell you:\n"
            "• The vast majority of moles are **not cancerous**\n"
            "• Melanoma detected early has a **98%+ survival rate**\n"
            "• YOLOCheck's result is a screening indicator, not a diagnosis\n\n"
            "Please book a dermatologist appointment — early detection saves lives. 💙"
        )

    if any(w in q for w in ["i am scared", "i'm scared", "i am worried",
                              "i'm worried", "so scared", "very scared"]):
        return (
            "I hear you, and feeling scared is completely understandable. 💙\n\n"
            "Please take a deep breath. YOLOCheck is not a diagnosis — it's an awareness tool. "
            "Many people with high risk scores go on to have completely benign results.\n\n"
            "The most important step: **see a dermatologist**. You are doing the right thing. 🌟"
        )

    if any(w in q for w in ["what should i do", "what do i do now", "help me", "what now"]):
        return (
            "**Here's exactly what to do:**\n\n"
            "1. **Don't panic** — most skin changes are benign and treatable\n"
            "2. **Book a dermatologist** — High Risk: 1 week, Moderate: 2–4 weeks, Low: annual\n"
            "3. **Document the mole** — take clear photos in good lighting\n"
            "4. **Monitor for changes** — growth, bleeding, itching, color change\n"
            "5. **Protect your skin** — SPF 50+ sunscreen daily\n\n"
            "You are taking the right steps by using YOLOCheck."
        )

    if any(w in q for w in ["stress", "mental health", "depressed", "can't sleep", "overthinking"]):
        return (
            "I hear you — health anxiety is very real. 💙\n\n"
            "The single best thing you can do for your mental health right now is to "
            "**book a dermatologist appointment**. Once you have a professional opinion, "
            "you will feel so much better.\n\n"
            "If anxiety is seriously affecting your daily life, please also speak with "
            "a mental health professional."
        )

    if any(w in q for w in ["embarrassed", "shy", "ashamed", "self conscious"]):
        return (
            "There is absolutely nothing to be embarrassed about. 💙\n\n"
            "Dermatologists see patients with all kinds of skin concerns every day. "
            "They are there to help, not to judge. Your health matters."
        )

    if any(w in q for w in ["can't afford", "no money", "expensive", "cost", "free"]):
        return (
            "**Accessing Skin Care on a Budget:**\n\n"
            "• Public hospitals — free or low-cost dermatology clinics\n"
            "• Medical colleges — teaching hospitals offer free consultations\n"
            "• Telemedicine — online consultations are often cheaper\n\n"
            "In Pakistan: PIMS, Jinnah Hospital, Aga Khan Hospital have dermatology departments.\n"
            "Many government hospitals offer subsidized consultations."
        )

    # ── Symptom questions ──────────────────────────────────────────────────────
    if any(w in q for w in ["itching", "itchy", "itch", "scratching"]):
        return (
            "**Itching Around a Mole:**\n\n"
            "Common causes (usually harmless): dry skin, irritation from clothing/soap.\n\n"
            "See a doctor if:\n"
            "• Itching persists more than 2 weeks\n"
            "• The mole has also changed in appearance\n"
            "• The mole is also bleeding\n\n"
            "Don't scratch — book a dermatologist appointment to be safe."
        )

    if any(w in q for w in ["bleeding", "bleed", "blood", "oozing"]):
        return (
            "**A Bleeding Mole — Take This Seriously:**\n\n"
            "1. Clean gently with mild soap and water\n"
            "2. Do NOT pick or scratch it\n"
            "3. Take a clear photo for reference\n"
            "4. Book a dermatologist appointment **within this week**\n\n"
            "Don't ignore a bleeding mole."
        )

    if any(w in q for w in ["changing", "changed", "growing", "getting bigger"]):
        return (
            "**A Changing Mole:**\n\n"
            "Any mole that changes in size, shape, or color should be evaluated.\n\n"
            "• Take photos now and compare monthly\n"
            "• Rapidly changing → dermatologist within 1 week\n"
            "• Slowly changing → appointment within 2–4 weeks"
        )

    if any(w in q for w in ["painful", "pain", "hurts", "sore", "tender"]):
        return (
            "**A Painful Mole:**\n\n"
            "Could be from trauma, infection, or rarely rapid growth.\n"
            "See a doctor if pain persists more than 2 weeks or mole is also changing."
        )

    if any(w in q for w in ["new mole", "new spot", "appeared", "suddenly appeared"]):
        return (
            "**A New Mole:**\n\n"
            "Usually harmless. More concerning if:\n"
            "• Appeared suddenly and grew quickly\n"
            "• Looks different from your other moles\n"
            "• Has irregular borders, multiple colors, or larger than 6mm\n\n"
            "Use YOLOCheck to scan it and see a dermatologist if it looks unusual."
        )

    if any(w in q for w in ["family history", "my family", "parent had",
                              "mother had", "father had", "hereditary", "genetic"]):
        return (
            "**Family History of Skin Cancer:**\n\n"
            "Increases your risk but does NOT mean you will develop it.\n\n"
            "• Full body skin check every 6–12 months\n"
            "• Monthly self-examinations\n"
            "• SPF 50+ every day\n"
            "• No tanning beds\n\n"
            "Tell your dermatologist about your family history."
        )

    if any(w in q for w in ["how long", "how much time", "will it spread", "fast", "quickly"]):
        return (
            "**How Fast Does Skin Cancer Progress?**\n\n"
            "• Basal cell carcinoma — very slow, rarely spreads\n"
            "• Squamous cell carcinoma — moderate, can spread if untreated\n"
            "• Melanoma — can spread quickly — early detection is critical\n\n"
            "Don't wait — book a dermatologist appointment now."
        )

    # ── Knowledge questions ────────────────────────────────────────────────────
    if ("benign" in q and "malignant" in q) or ("difference" in q and "mole" in q):
        return (
            "**Benign vs Malignant:**\n\n"
            "• **Benign** — non-cancerous, uniform, smooth, symmetric, stable\n"
            "• **Malignant** — cancerous, asymmetric, irregular, multiple colors\n\n"
            "Malignant can spread if untreated — early detection is key."
        )
    if "benign" in q:
        return (
            "**Benign Mole:**\n\n"
            "Non-cancerous. Round/oval, smooth borders, one color, symmetric, stable.\n"
            "Monitor regularly — generally no treatment needed."
        )
    if "malignant" in q:
        return (
            "**Malignant Mole:**\n\n"
            "Contains cancerous cells. Warning signs: asymmetry, irregular border, "
            "multiple colors, diameter >6mm, changing over time.\n"
            "See a dermatologist immediately if you notice these signs."
        )
    if "melanoma" in q:
        return (
            "**Melanoma:**\n\n"
            "Most serious skin cancer. 98%+ survival rate when detected early.\n"
            "Risk factors: UV exposure, fair skin, family history.\n"
            "Regular YOLOCheck scans and dermatologist visits are key."
        )
    if "basal" in q:
        return (
            "**Basal Cell Carcinoma:**\n\n"
            "Most common skin cancer. Grows slowly, rarely spreads.\n"
            "Highly treatable when caught early."
        )
    if "squamous" in q:
        return (
            "**Squamous Cell Carcinoma:**\n\n"
            "Second most common. Can spread if untreated.\n"
            "Treatable when detected early."
        )
    if "eczema" in q:
        return (
            "**Eczema:**\n\n"
            "Chronic inflammatory skin condition — dry, itchy, inflamed skin.\n"
            "Not contagious or cancerous. Managed with moisturizers and avoiding triggers."
        )
    if "psoriasis" in q:
        return (
            "**Psoriasis:**\n\n"
            "Chronic autoimmune condition — red patches with silvery scales.\n"
            "Not contagious. Treatment: topical creams and light therapy."
        )
    if "rosacea" in q:
        return (
            "**Rosacea:**\n\n"
            "Facial redness and flushing. Triggers: sun, hot drinks, stress.\n"
            "Manageable but not curable."
        )
    if "dermatitis" in q:
        return (
            "**Dermatitis:**\n\n"
            "Skin inflammation. Types: contact, atopic (eczema), seborrheic.\n"
            "Treatment: avoid triggers, topical treatments."
        )
    if "seborrheic" in q:
        return (
            "**Seborrheic Keratosis:**\n\n"
            "Common non-cancerous growth. Waxy, 'stuck on' appearance.\n"
            "Completely benign — no treatment needed unless irritated."
        )
    if "abcd" in q:
        return (
            "**The ABCD Rule:**\n\n"
            "• A — Asymmetry\n• B — Border irregularity\n"
            "• C — Color variation\n• D — Diameter >6mm\n\n"
            "TDS: 0–4 Low · 5–8 Moderate · 9–12 High Risk"
        )
    if "high risk" in q:
        return (
            "**High Risk:** Significant ABCD features detected.\n"
            "See a dermatologist within 1 week.\n"
            "NOT a diagnosis — many high risk scans turn out benign after professional evaluation."
        )
    if "low risk" in q:
        return "**Low Risk:** Features within normal range. Monthly self-checks + annual professional check."
    if "moderate risk" in q:
        return "**Moderate Risk:** Some concerning features. Book dermatologist within 2–4 weeks."
    if "prevent" in q or "prevention" in q:
        return (
            "**Preventing Skin Cancer:**\n\n"
            "• SPF 50+ sunscreen daily\n• Avoid sun 10am–4pm\n"
            "• No tanning beds\n• Monthly self-checks\n• Annual dermatologist visits"
        )
    if "dermatologist" in q or "doctor" in q or "when should" in q:
        return (
            "**When to See a Dermatologist:**\n\n"
            "• Immediately for High Risk\n• Within 2–4 weeks for Moderate Risk\n"
            "• Annually for routine checks\n"
            "• Any time a mole changes, bleeds, itches, or looks unusual"
        )
    if "spf" in q or "sunscreen" in q:
        return (
            "**Sunscreen:**\n\n"
            "• SPF 50+ broad-spectrum\n• Apply 15–30 min before sun\n"
            "• Reapply every 2 hours\n• Use year-round"
        )
    if "yolo" in q or "how does" in q or ("detect" in q and "mole" in q):
        return (
            "**How YOLOv11 Detects Moles:**\n\n"
            "YOLOv11 is a real-time neural network trained on dermoscopy images to:\n"
            "• Locate moles with precise bounding boxes\n"
            "• Classify as benign or malignant\n"
            "• Provide confidence scores\n\n"
            "ABCD analysis is then performed for additional risk assessment."
        )
    if "hello" in q or "hi" in q or "hey" in q:
        return (
            "Hello! I'm the YOLOCheck AI Health Assistant. 👋\n\n"
            "Ask me anything about skin health — moles, symptoms, diseases, "
            "your scan results, or even just how you're feeling. I'm here for you! 💙"
        )
    if "total" in q and "score" in q:
        return (
            "**Total Dermoscopy Score (TDS):**\n\n"
            "Sum of ABCD scores (each 0–3), total 0–12:\n"
            "• 0–4: Low Risk\n• 5–8: Moderate\n• 9–12: High Risk"
        )

    # ── Unknown question — try Gemini with simpler prompt ─────────────────────
    simple_system = (
        "You are a friendly, warm skin health assistant. Answer questions about "
        "skin health, moles, skin diseases, and emotional concerns in a clear, "
        "educational, and empathetic way. Never diagnose. Always suggest seeing "
        "a dermatologist for professional advice. Keep your answer concise and helpful."
    )
    gemini_reply = _call_gemini(message, simple_system, max_tokens=512)
    if gemini_reply:
        return gemini_reply

    # ── Final fallback ─────────────────────────────────────────────────────────
    return (
        "I'm here to help with any skin health questions! 💙\n\n"
        "You can ask me about:\n"
        "• Moles — benign vs malignant, ABCD rule, symptoms\n"
        "• Skin diseases — melanoma, eczema, psoriasis, rosacea\n"
        "• Your scan results — risk level, confidence, ABCD scores\n"
        "• Emotional support — if you're worried or scared\n"
        "• Prevention — sunscreen, self-checks, when to see a doctor\n\n"
        "What would you like to know?"
    )


def _fetch_scan_context(scan_id: str) -> str:
    try:
        supabase = get_supabase()
        scan     = supabase.table("scans").select("*").eq("id", scan_id).single().execute()
        if not scan.data:
            return ""
        s    = scan.data
        dets = (
            supabase.table("detections")
            .select("mole_id, risk_level, label, abcd_asymmetry, abcd_border, "
                    "abcd_color, abcd_diameter, abcd_total, confidence")
            .eq("scan_id", scan_id)
            .execute()
        )
        lines = [
            f"The user's scan detected {s['total_moles_detected']} mole(s).",
            f"Overall highest risk level: {s['highest_risk']}.",
        ]
        for d in (dets.data or []):
            lines.append(
                f"  • {d['mole_id']}: Label={d.get('label','unknown')}, "
                f"Risk={d['risk_level']}, "
                f"Confidence={round(d.get('confidence', 0) * 100)}%, "
                f"ABCD total={d['abcd_total']} "
                f"(A={d['abcd_asymmetry']}, B={d['abcd_border']}, "
                f"C={d['abcd_color']}, D={d['abcd_diameter']})"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Could not load scan context: %s", exc)
        return ""


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
        "contents": [{"parts": [{"text": full_prompt}], "role": "user"}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 1024},
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
        logger.warning("Gemini unavailable (%s) — using local fallback.", exc)
        return _local_fallback(message) + "\n\n" + DISCLAIMER