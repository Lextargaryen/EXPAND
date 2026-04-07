#Start_EXPAND

"""
╔══════════════════════════════════════════════════════════════╗
║          EXPAND — DARK FANTASY TEXT ADVENTURE ENGINE         ║
║                      Advanced v2.0                           ║
╚══════════════════════════════════════════════════════════════╝

Improvements over v1:
  - Full save/load system (JSON state file + human-readable session log)
  - Reality Grounding Layer: flags impossible actions BEFORE the LLM sees them
  - Player stats: HP, stamina, sanity
  - Persistent inventory system
  - Rolling conversation history for context continuity
  - Session transcript saved to .txt
  - Timestamped auto-save on every turn
"""

import gc
import json
import os
import random
import re
import requests
import sys
import torch
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

# ── PATHS ──────────────────────────────────────────────────────────────────────
SAVE_FILE      = "expand_save.json"
SESSION_LOG    = "expand_session_log.txt"
LORE_CACHE     = "world_lore_cache.md"

# ── VRAM CLEANUP ───────────────────────────────────────────────────────────────
print("Initializing engine and clearing VRAM...")
try:
    del trainer
except NameError:
    pass
torch.cuda.empty_cache()
gc.collect()
FastLanguageModel.for_inference(model)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — REALITY GROUNDING LAYER
# ══════════════════════════════════════════════════════════════════════════════

# These rules are checked LOCALLY (no LLM call needed) before passing to the GM.
# Each rule is a (pattern, grounding_hint) pair.
# The hint is injected into the GM prompt so it responds correctly.
IMPOSSIBLE_PATTERNS = [
    # Flight / levitation
    (r"\b(fly|flies|flying|levitat|float upward|soar|take off into the (sky|air))\b",
     "The player cannot fly. They have no wings, no magic flight ability, and gravity is absolute here. "
     "Deny this firmly but creatively — perhaps they jump and land hard, or look up enviously at something that can fly."),

    # Teleportation
    (r"\b(teleport|blink (to|away|across)|phase through|warp to|instantly appear)\b",
     "Teleportation does not exist in this world without a specific artifact. Deny it and describe them "
     "still standing where they were, perhaps disoriented from the attempt."),

    # Superhuman strength
    (r"\b(lift (the|a) (mountain|boulder|castle|tower|building)|rip (the|a) (wall|door|gate) (off|apart|out)|throw (the|a) (boulder|building|tower))\b",
     "This action exceeds human physical limits. The player cannot perform feats of impossible strength. "
     "Describe them straining uselessly, perhaps injuring themselves."),

    # Instant healing / resurrection
    (r"\b(instantly heal|fully heal|resurrect (myself|me)|come back (to life|from the dead)|un-die)\b",
     "There is no instant healing here. Wounds persist. Death is final unless a specific artifact or NPC "
     "allows otherwise. Describe the wound still aching, the blood still wet."),

    # Omniscience / meta-knowledge
    (r"\b(know everything|read (the (GM|game master|narrator)('s)? mind)|access (the|my) stats? (screen|sheet)|see (the|my) (inventory|map) screen)\b",
     "The player is not aware they are in a game. They experience this world as real. "
     "Respond from within the fiction — no menus, no screens, no GM mind-reading."),

    # God-mode invulnerability
    (r"\b(cannot be (hurt|killed|damaged|touched)|am invincible|turn invincible|become (immortal|invulnerable))\b",
     "No mortal is invincible. Deny this. The player is flesh and blood."),

    # Conjuring objects from nothing
    (r"\b(conjure (a |an )?(weapon|sword|shield|armor|food|water|item)|summon (a |an )?(weapon|sword|shield|item) from (thin air|nothing|nowhere))\b",
     "The player cannot conjure items they do not possess. They can only use what is in their inventory "
     "or what they find in the world."),

    # Time travel / reality reset
    (r"\b(go back in time|rewind|reset (the (world|game|time))|undo (that|what happened)|travel (back|forward) in time)\b",
     "Time does not bend to the player's will. What has happened has happened. "
     "Describe the cold finality of that."),
]

def check_reality(action: str) -> str | None:
    """
    Returns a grounding directive string if the action violates physical reality,
    or None if the action is plausible.
    """
    action_lower = action.lower()
    for pattern, hint in IMPOSSIBLE_PATTERNS:
        if re.search(pattern, action_lower):
            return hint
    return None


def classify_action_risk(action: str) -> str:
    """
    Loosely categorise how much a D20 roll matters for this action.
    Returns: 'risky' | 'social' | 'trivial'
    """
    risky_words   = r"\b(attack|fight|climb|sneak|pick (the )?lock|disarm|dodge|run|jump|force|break|steal|hack|stab|shoot|throw|wrestle)\b"
    social_words  = r"\b(persuade|convince|deceive|lie|intimidate|seduce|barter|negotiate|beg|threaten)\b"
    trivial_words = r"\b(look|examine|read|wait|sit|listen|smell|touch(?! the (beast|monster|creature))|pick up|take|drop|inventory|map)\b"

    action_lower = action.lower()
    if re.search(risky_words,  action_lower): return "risky"
    if re.search(social_words, action_lower): return "social"
    if re.search(trivial_words,action_lower): return "trivial"
    return "risky"  # default: assume risk

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — GAME STATE MANAGEMENT (SAVE / LOAD)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_STATS = {
    "hp":         20,
    "hp_max":     20,
    "stamina":    10,
    "stamina_max":10,
    "sanity":     10,
    "sanity_max": 10,
}

DEFAULT_PLAYER = {
    "name":      "Traveller",
    "stats":     DEFAULT_STATS.copy(),
    "inventory": [],
    "flags":     {},          # quest / story flags, e.g. {"met_aldric": True}
    "turn":      0,
}

DEFAULT_WORLD = {
    "current_location": "The Whispering Woods",
    "world_graph": {
        "The Whispering Woods": {
            "description": "A dark, oppressive forest where the fog seems to breathe and the trees lean inward as if listening.",
            "objects":     ["rusted longsword", "blood-stained map"],
            "connections": {},
            "visited":     True,
        }
    }
}


def new_game_state() -> dict:
    return {
        "player":    DEFAULT_PLAYER.copy(),
        "world":     {
            "current_location": DEFAULT_WORLD["current_location"],
            "world_graph":      {k: dict(v) for k, v in DEFAULT_WORLD["world_graph"].items()},
        },
        "history":   [],      # last N (action, response) pairs for context
        "created_at": datetime.now().isoformat(),
        "saved_at":   datetime.now().isoformat(),
    }


def save_game(state: dict) -> None:
    state["saved_at"] = datetime.now().isoformat()
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_game() -> dict | None:
    if not os.path.exists(SAVE_FILE):
        return None
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        print("[Warning] Save file corrupted. Starting fresh.")
        return None


def log_to_session(entry: str) -> None:
    """Append a line to the human-readable session transcript."""
    with open(SESSION_LOG, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def format_stats(player: dict) -> str:
    s = player["stats"]
    inv = ", ".join(player["inventory"]) if player["inventory"] else "nothing"
    return (
        f"HP {s['hp']}/{s['hp_max']} | "
        f"Stamina {s['stamina']}/{s['stamina_max']} | "
        f"Sanity {s['sanity']}/{s['sanity_max']} | "
        f"Inventory: {inv}"
    )


# ── Stat mutation helpers ──────────────────────────────────────────────────────

def apply_roll_to_stats(player: dict, roll: int, risk_type: str) -> str:
    """
    Mutate player stats based on roll result and risk type.
    Returns a terse damage/reward string injected into the GM prompt.
    """
    s = player["stats"]
    note = ""
    if risk_type == "trivial":
        return ""  # No stat consequence for trivial actions

    if roll <= 5:           # Critical failure
        if risk_type == "risky":
            dmg = random.randint(2, 5)
            s["hp"] = max(0, s["hp"] - dmg)
            note = f"[STAT CHANGE: Player took {dmg} HP damage. Current HP: {s['hp']}/{s['hp_max']}.]"
        elif risk_type == "social":
            s["sanity"] = max(0, s["sanity"] - 1)
            note = f"[STAT CHANGE: Player lost 1 Sanity. Current Sanity: {s['sanity']}/{s['sanity_max']}.]"
    elif 6 <= roll <= 9:    # Failure with partial cost
        if risk_type == "risky":
            dmg = random.randint(1, 3)
            s["hp"] = max(0, s["hp"] - dmg)
            sta = random.randint(1, 2)
            s["stamina"] = max(0, s["stamina"] - sta)
            note = (f"[STAT CHANGE: Player took {dmg} HP damage and lost {sta} Stamina. "
                    f"HP: {s['hp']}/{s['hp_max']}, Stamina: {s['stamina']}/{s['stamina_max']}.]")
    elif 10 <= roll <= 14:  # Mixed success
        if risk_type == "risky":
            sta = 1
            s["stamina"] = max(0, s["stamina"] - sta)
            note = f"[STAT CHANGE: Exertion cost 1 Stamina. Stamina: {s['stamina']}/{s['stamina_max']}.]"
    elif roll >= 18:        # Critical success — minor recovery
        s["stamina"] = min(s["stamina_max"], s["stamina"] + 1)
        note = f"[STAT CHANGE: Adrenaline restored 1 Stamina. Stamina: {s['stamina']}/{s['stamina_max']}.]"

    return note


def is_dead(player: dict) -> bool:
    return player["stats"]["hp"] <= 0


def is_insane(player: dict) -> bool:
    return player["stats"]["sanity"] <= 0

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — VECTOR RAG VAULT (LORE DATABASE)
# ══════════════════════════════════════════════════════════════════════════════

GITHUB_LORE_URL = "https://raw.githubusercontent.com/USERNAME/REPO/main/world_lore.md"

print("Loading lore database...")
raw_lore = ""

# Try cache first to save bandwidth
if os.path.exists(LORE_CACHE):
    with open(LORE_CACHE, "r", encoding="utf-8") as f:
        raw_lore = f.read()
    print(f"  Loaded lore from local cache ({LORE_CACHE}).")
else:
    try:
        response = requests.get(GITHUB_LORE_URL, timeout=10)
        response.raise_for_status()
        raw_lore = response.text
        with open(LORE_CACHE, "w", encoding="utf-8") as f:
            f.write(raw_lore)
        print("  Downloaded and cached lore from GitHub.")
    except Exception as e:
        print(f"  [Warning] Could not fetch lore: {e}. Using fallback.")
        raw_lore = (
            "The Whispering Woods are ancient and corrupted by the Unmaking.\n\n"
            "The Cult of Shadows seeks to dismantle the Stillstones and free the Outer Dark.\n\n"
            "Hollowmen were once human. Their torsos are hollow, filled with writhing shadow.\n\n"
            "The Stillheart is a fragment of a destroyed Stillstone that resists Unmaking."
        )

raw_chunks = [c.strip() for c in raw_lore.split("\n\n") if len(c.strip()) > 40]

print("  Building vector index...")
embedder        = SentenceTransformer("all-MiniLM-L6-v2")
lore_embeddings = embedder.encode(raw_chunks, show_progress_bar=False)
dimension       = lore_embeddings.shape[1]
vector_db       = faiss.IndexFlatL2(dimension)
vector_db.add(lore_embeddings.astype(np.float32))
print(f"  Vector index built. {len(raw_chunks)} lore chunks indexed.")


def get_relevant_lore(query: str, k: int = 2) -> str:
    if not raw_chunks:
        return "No lore available."
    q_vec              = embedder.encode([query]).astype(np.float32)
    distances, indices = vector_db.search(q_vec, min(k, len(raw_chunks)))
    return "\n".join(raw_chunks[i] for i in indices[0] if i < len(raw_chunks))

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PROMPTING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

GM_INSTRUCTION = """You are a strict, grounded, and imaginative Game Master running a dark fantasy text adventure set in the world of EXPAND — a world slowly being unmade by shadow.

═══ ABSOLUTE RULES ═══

1. PHYSICAL REALISM
   The player is a normal human. They cannot fly, teleport, conjure items, see through walls, or perform feats that defy physics. If a [REALITY VIOLATION] directive is present, you MUST enforce it. Deny the action creatively and immersively — describe what actually happens when they try and fail — but never simply say "you can't do that." Make the failure feel real and consequential.

2. D20 ROLL SYSTEM — NON-NEGOTIABLE
   A die roll is provided. You MUST obey it for any action tagged [RISKY] or [SOCIAL].
   [1–5]   → Critical Failure: severe consequence, possible injury or setback
   [6–9]   → Failure with partial cost: something goes wrong, but not catastrophically
   [10–14] → Mixed Success: success but with a complication or cost
   [15–19] → Full Success: clean outcome
   [20]    → Critical Success: exceptional result, bonus detail or discovery
   Trivial actions (looking, waiting, picking up items) do not require a roll result.

3. LORE INTEGRATION
   If [SECRET LORE] is provided, weave it into your response naturally. Never quote it directly — translate it into sensory, immersive prose.

4. MAP EXPANSION
   If the player discovers or enters a COMPLETELY NEW area not in the known map, you MUST end your response with this exact tag on its own line:
   [NEW_NODE] Location Name | item one, item two, item three

5. ITEM MANAGEMENT
   If the player picks up an item, end your response with:
   [PICKUP] item name
   If the player drops or loses an item:
   [DROP] item name

6. STAT EFFECTS
   If [STAT CHANGE] is provided, reflect it in your narration. A player at low HP should feel exhausted and desperate. A player at low Sanity should hear things, see things at the edges of their vision, struggle to trust their own perceptions.

7. PLAYER DEATH
   If [PLAYER IS DEAD] appears, narrate a final, complete death scene. Do not continue the adventure.

8. TONE
   Dark, literary, grounded. No comic relief unless the player initiates it. No exposition dumps. Show, don't tell. Sentences can be short and punishing.

9. RESPONSE LENGTH
   2–4 paragraphs per turn. No more. Leave space for the player to act.
"""


def build_prompt(state: dict, user_action: str, roll: int, risk_type: str,
                 reality_violation: str | None, stat_note: str) -> str:

    player   = state["player"]
    world    = state["world"]
    loc_name = world["current_location"]
    node     = world["world_graph"][loc_name]

    objects   = ", ".join(node["objects"]) if node["objects"] else "nothing of obvious interest"
    exits     = ", ".join(node["connections"].keys()) if node["connections"] else "none mapped yet"
    inventory = ", ".join(player["inventory"]) if player["inventory"] else "nothing"

    # Rolling context window: last 3 turns
    history_text = ""
    for turn in state["history"][-3:]:
        history_text += f'Player: {turn["action"]}\nGM: {turn["response"]}\n\n'

    # Build the context block
    ctx  = f"[LOCATION]\nName: {loc_name}\nDescription: {node['description']}\n"
    ctx += f"Objects here: {objects}\nKnown exits: {exits}\n\n"
    ctx += f"[PLAYER STATE]\nName: {player['name']}\nInventory: {inventory}\n"
    s    = player["stats"]
    ctx += (f"HP: {s['hp']}/{s['hp_max']} | "
            f"Stamina: {s['stamina']}/{s['stamina_max']} | "
            f"Sanity: {s['sanity']}/{s['sanity_max']}\n\n")

    ctx += f"[SECRET LORE]\n{get_relevant_lore(loc_name + ' ' + user_action)}\n\n"

    if history_text:
        ctx += f"[RECENT HISTORY]\n{history_text}"

    ctx += f"[CURRENT ACTION — tagged {risk_type.upper()}]\n"

    if is_dead(player):
        ctx += "[PLAYER IS DEAD] Narrate the death scene.\n"
    elif is_insane(player):
        ctx += "[PLAYER SANITY BROKEN] The player's mind has shattered. Their perception of reality is completely unreliable. Narrate accordingly.\n"

    if reality_violation:
        ctx += f"\n[REALITY VIOLATION DETECTED]\n{reality_violation}\n"
        ctx += "Enforce this violation. Do NOT allow the action to succeed or partially succeed.\n"
    else:
        ctx += f"[D20 ROLL: {roll}/20 — enforce the result for {risk_type} actions.]\n"
        if stat_note:
            ctx += f"{stat_note}\n"

    ctx += f"\nPlayer says: {user_action}"

    return ALPACA_PROMPT.format(GM_INSTRUCTION, ctx, "")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_response(raw: str, state: dict) -> tuple[str, bool]:
    """
    Parse the GM's raw response for system tags.
    Mutates state (world_graph, inventory) in place.
    Returns (clean_display_text, location_changed).
    """
    display   = raw
    loc_changed = False
    world     = state["world"]
    player    = state["player"]
    loc_name  = world["current_location"]

    # ── NEW_NODE ────────────────────────────────────────────────────────────
    if "[NEW_NODE]" in raw:
        parts   = raw.split("[NEW_NODE]")
        display = parts[0].strip()
        tag     = parts[1].strip()
        if "|" in tag:
            new_loc, items_raw = tag.split("|", 1)
            new_loc   = new_loc.strip()
            new_items = [i.strip() for i in items_raw.split(",") if i.strip()]

            world["world_graph"][new_loc] = {
                "description": f"An area discovered from {loc_name}.",
                "objects":     new_items,
                "connections": {"← Back": loc_name},
                "visited":     True,
            }
            world["world_graph"][loc_name]["connections"][new_loc] = new_loc
            world["current_location"] = new_loc
            loc_changed = True
            print(f"\n  [Map expanded → {new_loc}]")

    # ── PICKUP ──────────────────────────────────────────────────────────────
    for pickup_match in re.finditer(r"\[PICKUP\]\s*(.+)", display):
        item      = pickup_match.group(1).strip()
        node      = world["world_graph"][world["current_location"]]
        # Remove from location if present
        node["objects"] = [o for o in node["objects"] if o.lower() != item.lower()]
        if item not in player["inventory"]:
            player["inventory"].append(item)
        print(f"  [Inventory: +{item}]")
    display = re.sub(r"\[PICKUP\]\s*.+", "", display).strip()

    # ── DROP ────────────────────────────────────────────────────────────────
    for drop_match in re.finditer(r"\[DROP\]\s*(.+)", display):
        item = drop_match.group(1).strip()
        player["inventory"] = [i for i in player["inventory"] if i.lower() != item.lower()]
        node = world["world_graph"][world["current_location"]]
        node["objects"].append(item)
        print(f"  [Inventory: -{item}]")
    display = re.sub(r"\[DROP\]\s*.+", "", display).strip()

    return display.strip(), loc_changed

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BUILT-IN COMMANDS (handled before LLM call)
# ══════════════════════════════════════════════════════════════════════════════

def handle_builtin(cmd: str, state: dict) -> bool:
    """
    Handle meta-commands locally. Returns True if handled (skip LLM call).
    """
    cmd_l = cmd.strip().lower()

    if cmd_l in ("inventory", "i", "inv"):
        inv = ", ".join(state["player"]["inventory"]) or "You are carrying nothing."
        print(f"\nInventory: {inv}")
        return True

    if cmd_l in ("stats", "status", "s"):
        print(f"\n{format_stats(state['player'])}")
        return True

    if cmd_l in ("map", "m"):
        print("\n[Known Locations]")
        for name, data in state["world"]["world_graph"].items():
            marker = "► " if name == state["world"]["current_location"] else "  "
            exits  = ", ".join(data["connections"].keys()) or "none"
            print(f"{marker}{name}  [exits: {exits}]")
        return True

    if cmd_l in ("save",):
        save_game(state)
        print("\n[Game saved.]")
        return True

    if cmd_l.startswith("name "):
        state["player"]["name"] = cmd.strip()[5:].strip()
        print(f"\n[Name set to: {state['player']['name']}]")
        return True

    if cmd_l in ("help", "?", "h"):
        print("""
[Built-in commands]
  inventory / i       — Show inventory
  stats / s           — Show HP, stamina, sanity
  map / m             — Show discovered locations
  save                — Save game to file
  name <your name>    — Set your character name
  help / ?            — This help text
  quit / exit         — End session
        """)
        return True

    return False

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run():
    # ── Load or new game ───────────────────────────────────────────────────
    state = None
    if os.path.exists(SAVE_FILE):
        answer = input("Save file found. Load it? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            state = load_game()
            if state:
                print(f"  Loaded save from {state.get('saved_at', '?')}")

    if state is None:
        print("Starting new game...")
        state = new_game_state()
        player_name = input("Enter your character's name (or press Enter for 'Traveller'): ").strip()
        if player_name:
            state["player"]["name"] = player_name
        save_game(state)

    # ── Session log header ─────────────────────────────────────────────────
    log_to_session("\n" + "═"*60)
    log_to_session(f"SESSION START — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_session(f"Character: {state['player']['name']}")
    log_to_session("═"*60 + "\n")

    # ── Welcome ────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EXPAND — A Dark Fantasy Text Adventure")
    print("═"*60)
    print(f"  Welcome, {state['player']['name']}.")
    print(f"  {format_stats(state['player'])}")
    loc = state["world"]["current_location"]
    node = state["world"]["world_graph"][loc]
    print(f"\n  You are in: {loc}")
    print(f"  {node['description']}\n")
    print("  Type 'help' for commands. Type 'quit' to exit.")
    print("═"*60 + "\n")

    # ── Main loop ──────────────────────────────────────────────────────────
    while True:
        loc_name = state["world"]["current_location"]
        try:
            user_action = input(f"[{loc_name}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session interrupted. Saving...]")
            save_game(state)
            break

        if not user_action:
            continue

        if user_action.lower() in ("quit", "exit"):
            save_game(state)
            print("\n[Game saved. Farewell.]\n")
            log_to_session("\n[SESSION ENDED]\n")
            break

        # ── Built-in commands ──────────────────────────────────────────────
        if handle_builtin(user_action, state):
            continue

        # ── Reality check (BEFORE LLM) ─────────────────────────────────────
        reality_violation = check_reality(user_action)

        # ── Roll & risk ────────────────────────────────────────────────────
        risk_type = classify_action_risk(user_action)
        if reality_violation:
            roll = 0   # Roll is irrelevant for impossible actions
        else:
            roll = random.randint(1, 20)

        # ── Stat consequences ──────────────────────────────────────────────
        stat_note = ""
        if not reality_violation:
            stat_note = apply_roll_to_stats(state["player"], roll, risk_type)

        # ── Build & send prompt ────────────────────────────────────────────
        prompt = build_prompt(state, user_action, roll, risk_type, reality_violation, stat_note)

        inputs  = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 420,
            use_cache          = True,
            temperature        = 0.75,
            repetition_penalty = 1.15,
        )
        raw_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        raw_response = raw_output.split("### Response:\n")[-1].strip()

        # ── Parse GM response ──────────────────────────────────────────────
        display_text, loc_changed = parse_response(raw_response, state)

        # ── Roll indicator ─────────────────────────────────────────────────
        if reality_violation:
            roll_label = "[REALITY VIOLATION — roll ignored]"
        elif risk_type == "trivial":
            roll_label = "[Trivial action — no roll]"
        else:
            roll_label = f"[D20: {roll}/20 — {risk_type}]"

        # ── Print ──────────────────────────────────────────────────────────
        print(f"\n  {roll_label}")
        print(f"  {format_stats(state['player'])}")
        print(f"\nGM: {display_text}\n")
        print("-" * 60)

        # ── Update state ───────────────────────────────────────────────────
        state["player"]["turn"] += 1
        state["history"].append({
            "turn":     state["player"]["turn"],
            "action":   user_action,
            "roll":     roll,
            "risk":     risk_type,
            "response": display_text,
        })
        # Keep history from growing unbounded
        if len(state["history"]) > 20:
            state["history"] = state["history"][-20:]

        # ── Session log ────────────────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        log_to_session(f"[{ts}] Turn {state['player']['turn']} | {roll_label}")
        log_to_session(f"Player: {user_action}")
        log_to_session(f"GM: {display_text}")
        log_to_session(f"{format_stats(state['player'])}\n")

        # ── Auto-save every turn ───────────────────────────────────────────
        save_game(state)

        # ── Death / insanity check ─────────────────────────────────────────
        if is_dead(state["player"]):
            print("\n═══════════════════════════════════════")
            print("  YOUR STORY ENDS HERE.")
            print("═══════════════════════════════════════\n")
            log_to_session("[PLAYER DIED]\n")
            answer = input("Start a new game? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                # Archive the old save
                if os.path.exists(SAVE_FILE):
                    archive = SAVE_FILE.replace(".json", f"_dead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    os.rename(SAVE_FILE, archive)
                state = new_game_state()
                save_game(state)
                print("New game started.\n")
            else:
                break

        elif is_insane(state["player"]):
            print("\n  [Your sanity is gone. The world has become... something else.]\n")


if __name__ == "__main__":
    run()