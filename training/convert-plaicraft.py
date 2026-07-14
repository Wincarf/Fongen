#!/usr/bin/env python3
"""
convert-plaicraft.py — Convert PLAICraft keyboard/mouse data to Fongen training examples.

Takes PLAICraft SQLite databases (keyboard/mouse events) and produces
JSONL training examples in the Fongen action/planning schema.

Usage:
  python convert-plaicraft.py --input ./plaicraft-data/ --output ./training-data/
  python convert-plaicraft.py --input session.db --output examples.jsonl --window 500
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Key name -> action mapping (Minecraft 1.19.4 defaults)
KEY_TO_ACTION: dict[str, str] = {
    "key.forward": "MOVE_FORWARD",
    "key.keyboard.w": "MOVE_FORWARD",
    "w": "MOVE_FORWARD",
    "key.back": "MOVE_BACK",
    "key.keyboard.s": "MOVE_BACK",
    "s": "MOVE_BACK",
    "key.left": "TURN_LEFT",
    "key.keyboard.a": "TURN_LEFT",
    "a": "TURN_LEFT",
    "key.right": "TURN_RIGHT",
    "key.keyboard.d": "TURN_RIGHT",
    "d": "TURN_RIGHT",
    "key.jump": "JUMP",
    "key.keyboard.space": "JUMP",
    "space": "JUMP",
    "key.sprint": "SPRINT",
    "key.keyboard.left.control": "SPRINT",
    "ctrl": "SPRINT",
    "key.sneak": "SNEAK",
    "key.keyboard.left.shift": "SNEAK",
    "shift": "SNEAK",
    "key.use": "USE_ITEM",
    "key.mouse.right": "USE_ITEM",
    "key.attack": "MINE_TARGET",
    "key.mouse.left": "MINE_TARGET",
    "key.pickItem": "PLACE_BLOCK",
    "key.drop": "STOP",
    "key.inventory": "WAIT",
    "key.keyboard.e": "WAIT",
    "e": "WAIT",
}

VALID_ACTIONS = [
    "MOVE_FORWARD", "MOVE_BACK", "TURN_LEFT", "TURN_RIGHT",
    "LOOK_UP", "LOOK_DOWN", "JUMP", "SPRINT", "SNEAK",
    "MINE_TARGET", "PLACE_BLOCK", "ATTACK", "USE_ITEM",
    "EAT", "STOP", "WAIT", "REQUEST_PLAN",
]

INTENT_MAP: dict[str, str] = {
    "MINE_TARGET": "MINE_TASK",
    "MOVE_FORWARD": "GOTO",
    "MOVE_BACK": "GOTO",
    "USE_ITEM": "PLACE_TASK",
    "ATTACK": "SURVIVE",
    "WAIT": "IDLE",
    "STOP": "IDLE",
}


def discover_tables(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def discover_keymouse_tables(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    """Find keyboard and mouse event tables by inspecting schema."""
    tables = discover_tables(conn)
    keyboard_table = None
    mouse_table = None

    for table in tables:
        cols = [
            row[1]
            for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        ]
        col_lower = [c.lower() for c in cols]
        has_timestamp = any("time" in c or "ts" in c for c in col_lower)
        has_key = any("key" in c for c in col_lower)
        has_button = any("button" in c or "mouse" in c for c in col_lower)
        has_action = any("action" in c or "event" in c or "type" in c for c in col_lower)

        if has_timestamp and has_key and (has_action or "press" in str(cols).lower()):
            if keyboard_table is None:
                keyboard_table = table
        elif has_timestamp and has_button:
            if mouse_table is None:
                mouse_table = table

    return keyboard_table, mouse_table


def load_events(
    conn: sqlite3.Connection,
    table: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = f"SELECT * FROM '{table}'"
    if limit:
        query += f" LIMIT {limit}"
    cursor = conn.execute(query)
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def find_timestamp_field(event: dict[str, Any]) -> float:
    for key in ("timestamp", "ts", "time", "event_time", "t"):
        if key in event:
            val = event[key]
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return 0.0


def find_key_field(event: dict[str, Any]) -> str:
    for key in ("key", "key_name", "keyname", "key_code", "keycode", "button", "name"):
        if key in event:
            return str(event[key])
    return ""


def find_action_field(event: dict[str, Any]) -> str:
    for key in ("action", "event", "type", "event_type", "state"):
        if key in event:
            val = str(event[key]).lower()
            if val in ("press", "pressed", "down", "1", "true"):
                return "press"
            if val in ("release", "released", "up", "0", "false"):
                return "release"
            return val
    return "press"


def map_to_action(key_name: str, action: str, mouse_dx: float = 0, mouse_dy: float = 0) -> str:
    """Map a key/mouse event to a Fongen action."""
    if action != "press":
        return "STOP"

    key_lower = key_name.lower().strip()

    # Direct key mapping
    for pattern, mapped in KEY_TO_ACTION.items():
        if pattern in key_lower or key_lower in pattern:
            return mapped

    # Mouse movement -> look/turn
    if abs(mouse_dy) > abs(mouse_dx) and abs(mouse_dy) > 5:
        return "LOOK_DOWN" if mouse_dy > 0 else "LOOK_UP"
    if abs(mouse_dx) > 5:
        return "TURN_RIGHT" if mouse_dx > 0 else "TURN_LEFT"

    return "WAIT"


def window_events(
    events: list[dict[str, Any]],
    window_ms: int,
) -> list[list[dict[str, Any]]]:
    """Group events into time windows."""
    if not events:
        return []
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    window_start = find_timestamp_field(events[0])

    for event in events:
        ts = find_timestamp_field(event)
        if ts - window_start >= window_ms:
            if current:
                windows.append(current)
            current = []
            window_start = ts
        current.append(event)

    if current:
        windows.append(current)
    return windows


def dominant_action(events: list[dict[str, Any]]) -> str:
    """Determine the dominant action in a window of events."""
    if not events:
        return "WAIT"

    actions: list[str] = []
    for event in events:
        key = find_key_field(event)
        action_type = find_action_field(event)
        dx = float(event.get("dx", event.get("x_delta", 0)) or 0)
        dy = float(event.get("dy", event.get("y_delta", 0)) or 0)
        actions.append(map_to_action(key, action_type, dx, dy))

    counter = Counter(actions)
    # Filter out STOP and WAIT for dominance
    significant = {a: c for a, c in counter.items() if a not in ("STOP", "WAIT")}
    if significant:
        return max(significant, key=significant.get)
    return counter.most_common(1)[0][0]


def create_training_example(
    window_events_list: list[dict[str, Any]],
    prev_actions: list[str],
    window_idx: int,
) -> dict[str, Any]:
    """Create a single training example from a window of events."""
    action = dominant_action(window_events_list)

    # Build a compact context description
    context_parts: list[str] = []
    if prev_actions:
        recent = ", ".join(prev_actions[-5:])
        context_parts.append(f"Recent actions: {recent}")
    context_parts.append(f"Window index: {window_idx}")

    # High-level intent inference
    intent = INTENT_MAP.get(action, "IDLE")

    # Low-level action example
    action_example = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Fongen, an embodied Minecraft agent. "
                    "Given the recent context, predict the next action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "recent_actions": prev_actions[-10:],
                    "current_action_context": context_parts[-1],
                    "available_actions": VALID_ACTIONS,
                }),
            },
            {
                "role": "assistant",
                "content": json.dumps({"action": action, "reason": "imitation from PLAICraft"}),
            },
        ],
    }

    return action_example


def create_planning_example(
    action_window: list[str],
    window_idx: int,
) -> dict[str, Any]:
    """Create a high-level planning example from a sequence of actions."""
    if not action_window:
        return {}

    # Determine dominant intent from action sequence
    counter = Counter(action_window)
    dominant = counter.most_common(1)[0][0]
    intent = INTENT_MAP.get(dominant, "IDLE")

    goal: dict[str, Any] = {
        "intent": intent,
        "priority": "normal",
        "reason": f"Inferred from action sequence: {action_window[:10]}",
    }
    if intent == "MINE_TASK":
        goal["target"] = "minecraft:stone"
        goal["count"] = min(counter.get("MINE_TARGET", 1), 8)
    elif intent == "GOTO":
        goal["target"] = "explore"

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Fongen, an embodied Minecraft agent. "
                    "Analyze the situation and choose the next goal. "
                    "Respond with strict JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "observation": {
                        "health": 20,
                        "food": 20,
                        "position": {"x": 0, "y": 64, "z": 0},
                        "in_danger": False,
                        "recent_actions": action_window[:10],
                    },
                    "available_intents": [
                        "GOTO", "MINE_TASK", "CRAFT_TASK", "PLACE_TASK",
                        "FOLLOW_PLAYER", "SURVIVE", "IDLE",
                    ],
                }),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "thought": f"Based on recent actions ({dominant}), continuing {intent.lower()}",
                    "goal": goal,
                    "confidence": 0.6,
                }),
            },
        ],
    }


def process_sqlite(
    db_path: str,
    output_dir: Path,
    window_ms: int = 500,
    max_examples: int = 500,
) -> tuple[int, int]:
    """Process a single PLAICraft SQLite file."""
    conn = sqlite3.connect(db_path)
    try:
        kb_table, mouse_table = discover_keymouse_tables(conn)
        if not kb_table and not mouse_table:
            print(f"  [SKIP] No keyboard/mouse tables found in {db_path}")
            return 0, 0

        all_events: list[dict[str, Any]] = []
        if kb_table:
            print(f"  [INFO] Loading keyboard events from '{kb_table}'...")
            all_events.extend(load_events(conn, kb_table))
        if mouse_table:
            print(f"  [INFO] Loading mouse events from '{mouse_table}'...")
            all_events.extend(load_events(conn, mouse_table))

        if not all_events:
            print(f"  [SKIP] No events found in {db_path}")
            return 0, 0

        # Sort by timestamp
        all_events.sort(key=find_timestamp_field)

        # Window the events
        windows = window_events(all_events, window_ms)
        print(f"  [INFO] {len(windows)} windows from {len(all_events)} events")

        action_examples: list[dict[str, Any]] = []
        planning_examples: list[dict[str, Any]] = []
        prev_actions: list[str] = []
        action_seq: list[str] = []

        for idx, window in enumerate(windows[:max_examples * 2]):
            ex = create_training_example(window, prev_actions, idx)
            action_examples.append(ex)

            action = dominant_action(window)
            prev_actions.append(action)
            action_seq.append(action)

            # Create planning example every 10 windows
            if len(action_seq) >= 10:
                plan_ex = create_planning_example(action_seq, idx)
                if plan_ex:
                    planning_examples.append(plan_ex)
                action_seq = []

            if len(action_examples) >= max_examples:
                break

        # Write outputs
        session_name = Path(db_path).stem
        action_file = output_dir / f"{session_name}_actions.jsonl"
        plan_file = output_dir / f"{session_name}_planning.jsonl"

        with open(action_file, "w") as f:
            for ex in action_examples:
                f.write(json.dumps(ex) + "\n")

        with open(plan_file, "w") as f:
            for ex in planning_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"  [OK] {len(action_examples)} action + {len(planning_examples)} planning examples -> {output_dir}")
        return len(action_examples), len(planning_examples)

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Convert PLAICraft data to Fongen training examples")
    parser.add_argument("--input", required=True, help="Input directory or single .db file")
    parser.add_argument("--output", default="./training-data", help="Output directory for JSONL files")
    parser.add_argument("--window", type=int, default=500, help="Window size in ms (default: 500)")
    parser.add_argument("--max-examples", type=int, default=500, help="Max examples per session (default: 500)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix in (".db", ".sqlite", ".sqlite3"):
        process_sqlite(str(input_path), output_dir, args.window, args.max_examples)
    elif input_path.is_dir():
        db_files = list(input_path.rglob("*.db")) + list(input_path.rglob("*.sqlite")) + list(input_path.rglob("*.sqlite3"))
        if not db_files:
            print(f"[ERROR] No .db/.sqlite files found in {input_path}")
            sys.exit(1)

        total_actions = 0
        total_planning = 0
        for db_file in db_files:
            print(f"\n[PROCESS] {db_file}")
            a, p = process_sqlite(str(db_file), output_dir, args.window, args.max_examples)
            total_actions += a
            total_planning += p

        print(f"\n[DONE] Total: {total_actions} action examples, {total_planning} planning examples")
        print(f"[INFO] Output: {output_dir}/")
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
