#!/usr/bin/env python3
"""
generate-synthetic.py — Generate synthetic Fongen training examples.

Fallback when PLAICraft conversion is too slow. Generates rule-based
state->goal and state->action examples in the same JSONL format.

Usage:
  python generate-synthetic.py --output ./training-data/ --count 2000
"""
import argparse
import json
import random
from pathlib import Path

INTENTS = ["GOTO", "MINE_TASK", "CRAFT_TASK", "PLACE_TASK", "FOLLOW_PLAYER", "SURVIVE", "IDLE"]

BLOCK_TYPES = [
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:stone", "minecraft:cobblestone", "minecraft:coal_ore",
    "minecraft:iron_ore", "minecraft:gold_ore", "minecraft:diamond_ore",
    "minecraft:dirt", "minecraft:sand", "minecraft:gravel",
]

CRAFTABLE_ITEMS = [
    "minecraft:oak_planks", "minecraft:crafting_table", "minecraft:stick",
    "minecraft:stone_pickaxe", "minecraft:iron_pickaxe", "minecraft:torch",
    "minecraft:bread", "minecraft:wooden_sword",
]

BIOMES = ["plains", "forest", "birch_forest", "desert", "mountains", "swamp", "jungle", "taiga", "beach", "savanna"]

FOOD_ITEMS = ["bread", "cooked_beef", "cooked_porkchop", "apple", "carrot", "cooked_cod", "baked_potato"]

EQUIPPABLE = ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "wooden_axe", "stone_axe",
              "iron_axe", "wooden_sword", "stone_sword", "iron_sword", "empty", "bread", "apple"]

HOSTILE_MOBS = ["zombie", "skeleton", "creeper", "spider", "enderman", "witch"]
PASSIVE_MOBS = ["cow", "pig", "sheep", "chicken", "horse", "rabbit"]


def random_position():
    return {
        "x": round(random.uniform(-500, 500), 1),
        "y": round(random.uniform(60, 100), 1),
        "z": round(random.uniform(-500, 500), 1),
    }


def random_inventory():
    items = []
    num_items = random.randint(1, 8)
    for _ in range(num_items):
        item_type = random.choice(BLOCK_TYPES + CRAFTABLE_ITEMS + FOOD_ITEMS)
        item_name = item_type.replace("minecraft:", "")
        items.append({
            "name": item_name,
            "count": random.randint(1, 64),
            "displayName": item_name.replace("_", " ").title(),
        })
    return items


def random_nearby_blocks():
    blocks = []
    num = random.randint(3, 10)
    for _ in range(num):
        block = random.choice(BLOCK_TYPES).replace("minecraft:", "")
        blocks.append({
            "name": block,
            "count": random.randint(1, 50),
            "closest_distance": round(random.uniform(1, 16), 1),
            "direction": random.choice(["north", "south", "east", "west", "up", "down", "nearby"]),
        })
    return blocks


def random_nearby_entities():
    entities = []
    num = random.randint(0, 4)
    for _ in range(num):
        if random.random() < 0.3:
            name = random.choice(HOSTILE_MOBS)
            is_hostile = True
            etype = "mob"
        else:
            name = random.choice(PASSIVE_MOBS)
            is_hostile = False
            etype = "animal"
        pos = random_position()
        entities.append({
            "name": name,
            "type": etype,
            "position": pos,
            "distance": round(random.uniform(2, 30), 1),
            "is_hostile": is_hostile,
        })
    return entities


def generate_observation(force_danger=False, force_low_health=False, force_hungry=False):
    health = random.uniform(1, 20) if force_low_health else random.uniform(10, 20)
    food = random.uniform(0, 6) if force_hungry else random.uniform(10, 20)
    entities = random_nearby_entities()

    has_hostile = any(e["is_hostile"] and e["distance"] < 16 for e in entities)
    is_danger = force_danger or has_hostile or health < 10

    return {
        "timestamp": 0,
        "health": round(health, 1),
        "food": round(food, 1),
        "saturation": round(random.uniform(0, 20), 1),
        "position": random_position(),
        "biome": random.choice(BIOMES),
        "time_of_day": random.choice(["day", "night", "dusk", "dawn"]),
        "is_in_danger": is_danger,
        "equipped_item": random.choice(EQUIPPABLE),
        "inventory_summary": random_inventory(),
        "nearby_blocks": random_nearby_blocks(),
        "nearby_entities": entities,
        "recent_events": [],
    }


def decide_goal(observation) -> dict:
    """Rule-based goal selection that the model should learn."""
    if observation["is_in_danger"] or observation["health"] < 10:
        return {
            "thought": "Danger detected or low health, switching to survival mode",
            "goal": {
                "intent": "SURVIVE",
                "priority": "critical",
                "reason": "Health critical or hostile mob nearby",
            },
            "confidence": 0.9,
        }

    if observation["food"] < 6:
        return {
            "thought": "Food is low, need to eat or find food",
            "goal": {
                "intent": "SURVIVE",
                "priority": "high",
                "reason": "Hunger level dangerously low",
            },
            "confidence": 0.85,
        }

    # Check for wood/logs nearby
    wood_blocks = [b for b in observation["nearby_blocks"] if "log" in b["name"]]
    if wood_blocks and observation["food"] > 12:
        wood = wood_blocks[0]
        has_axe = any("axe" in i["name"] for i in observation["inventory_summary"])
        return {
            "thought": f"Found {wood['name']} nearby, collecting wood for crafting",
            "goal": {
                "intent": "MINE_TASK",
                "target": f"minecraft:{wood['name']}",
                "count": random.choice([4, 8, 12]),
                "priority": "normal",
                "reason": "Need wood for planks and tools",
            },
            "confidence": 0.8 if has_axe else 0.5,
        }

    # Check for ores
    ore_blocks = [b for b in observation["nearby_blocks"] if "ore" in b["name"]]
    if ore_blocks and any("pickaxe" in i["name"] for i in observation["inventory_summary"]):
        ore = ore_blocks[0]
        return {
            "thought": f"Found {ore['name']} nearby with a pickaxe equipped",
            "goal": {
                "intent": "MINE_TASK",
                "target": f"minecraft:{ore['name']}",
                "count": random.choice([4, 8, 16]),
                "priority": "high" if "diamond" in ore["name"] or "gold" in ore["name"] else "normal",
                "reason": f"Valuable ore: {ore['name']}",
            },
            "confidence": 0.75,
        }

    # Night time without shelter
    if observation["time_of_day"] == "night":
        return {
            "thought": "It is nighttime, should seek shelter or stay safe",
            "goal": {
                "intent": "SURVIVE",
                "priority": "high",
                "reason": "Night is dangerous with mobs",
            },
            "confidence": 0.7,
        }

    # Has wood but no planks
    has_logs = any("log" in i["name"] for i in observation["inventory_summary"])
    has_planks = any("planks" in i["name"] for i in observation["inventory_summary"])
    if has_logs and not has_planks:
        return {
            "thought": "Have logs but no planks, should craft",
            "goal": {
                "intent": "CRAFT_TASK",
                "target": "minecraft:oak_planks",
                "count": 4,
                "priority": "normal",
                "reason": "Convert logs to planks for building",
            },
            "confidence": 0.7,
        }

    # Has planks but no crafting table
    has_planks = any("planks" in i["name"] for i in observation["inventory_summary"])
    has_table = any("crafting_table" in i["name"] for i in observation["inventory_summary"])
    if has_planks and not has_table:
        return {
            "thought": "Have planks but no crafting table, need to make one",
            "goal": {
                "intent": "CRAFT_TASK",
                "target": "minecraft:crafting_table",
                "count": 1,
                "priority": "normal",
                "reason": "Need crafting table for advanced recipes",
            },
            "confidence": 0.7,
        }

    # Default: explore
    target_pos = observation["position"]
    direction = random.choice([(20, 0), (-20, 0), (0, 20), (0, -20)])
    return {
        "thought": "No immediate needs, exploring the environment for resources",
        "goal": {
            "intent": "GOTO",
            "position": {
                "x": target_pos["x"] + direction[0],
                "y": target_pos["y"],
                "z": target_pos["z"] + direction[1],
            },
            "priority": "low",
            "reason": "Exploring for new resources",
        },
        "confidence": 0.4,
    }


def create_planning_example(observation, goal_response):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Fongen, an embodied AI agent operating inside Minecraft "
                    "through a Mineflayer body. Observe the world state, reason about "
                    "the situation, and choose the next goal. Respond with strict JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "health": observation["health"],
                    "food": observation["food"],
                    "position": observation["position"],
                    "biome": observation["biome"],
                    "time": observation["time_of_day"],
                    "in_danger": observation["is_in_danger"],
                    "equipped": observation["equipped_item"],
                    "inventory": [f"{i['count']}x {i['name']}" for i in observation["inventory_summary"]],
                    "nearby_blocks": [
                        {"name": b["name"], "dist": b["closest_distance"], "dir": b["direction"]}
                        for b in observation["nearby_blocks"][:8]
                    ],
                    "nearby_entities": [
                        {"name": e["name"], "type": e["type"], "dist": e["distance"], "hostile": e["is_hostile"]}
                        for e in observation["nearby_entities"][:5]
                    ],
                }),
            },
            {
                "role": "assistant",
                "content": json.dumps(goal_response),
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Fongen training data")
    parser.add_argument("--output", default="./training-data", help="Output directory")
    parser.add_argument("--count", type=int, default=2000, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    scenarios = [
        ("normal", 0.5),
        ("danger", 0.15),
        ("low_health", 0.1),
        ("hungry", 0.1),
        ("night", 0.15),
    ]

    for i in range(args.count):
        # Weighted scenario selection
        r = random.random()
        cumulative = 0
        scenario = "normal"
        for name, weight in scenarios:
            cumulative += weight
            if r < cumulative:
                scenario = name
                break

        force_danger = scenario == "danger"
        force_low_health = scenario == "low_health"
        force_hungry = scenario == "hungry"
        obs = generate_observation(force_danger, force_low_health, force_hungry)

        if scenario == "night":
            obs["time_of_day"] = "night"

        goal = decide_goal(obs)
        example = create_planning_example(obs, goal)
        examples.append(example)

    output_file = output_dir / "synthetic_planning.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[DONE] Generated {len(examples)} synthetic planning examples")
    print(f"[INFO] Output: {output_file}")
    print(f"[INFO] Scenarios: {dict(scenarios)}")


if __name__ == "__main__":
    main()
