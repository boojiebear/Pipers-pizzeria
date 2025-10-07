"""
clash_bot_arena.py
Clash Royale–style simulator + MCTS bot + arena visualization using matplotlib.
"""

import math
import random
import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# -------------------------
# Simplified Game Model
# -------------------------

class Card:
    def __init__(self, name: str, cost: int, damage: int):
        self.name = name
        self.cost = cost
        self.damage = damage

    def __repr__(self):
        return f"{self.name}(cost={self.cost}, dmg={self.damage})"


class SimpleClashGame:
    """
    Very simplified 1v1 Clash Royale-like game.
    - Each player has 10 elixir max, regenerates 1 elixir per turn.
    - Players take turns playing one card per step (if they can afford it).
    - Card deals tower damage directly (no troop simulation).
    - Game ends if one tower <= 0 or after max_turns.
    """

    def __init__(self, starting_hp=100, max_turns=50):
        self.starting_hp = starting_hp
        self.max_turns = max_turns

        # State
        self.turn = 0
        self.current_player = 0  # 0 = bot, 1 = opponent
        self.hp = [starting_hp, starting_hp]
        self.elixir = [5, 5]  # start with 5
        self.hands: List[List[Card]] = [
            [
                Card("Goblin", 2, 5),
                Card("Knight", 3, 8),
                Card("Giant", 5, 15),
            ],
            [
                Card("Goblin", 2, 5),
                Card("Knight", 3, 8),
                Card("Giant", 5, 15),
            ],
        ]
        self.winner: Optional[int] = None
        self.arena_cards: List[Dict] = []  # cards moving on the arena

    def clone(self):
        g = SimpleClashGame(self.starting_hp, self.max_turns)
        g.turn = self.turn
        g.current_player = self.current_player
        g.hp = self.hp[:]
        g.elixir = self.elixir[:]
        g.hands = [[Card(c.name, c.cost, c.damage) for c in hand] for hand in self.hands]
        g.winner = self.winner
        g.arena_cards = [c.copy() for c in self.arena_cards]
        return g

    def get_state(self) -> Dict:
        return {
            "turn": self.turn,
            "current_player": self.current_player,
            "hp": self.hp[:],
            "elixir": self.elixir[:],
            "hand": [{"name": c.name, "cost": c.cost, "damage": c.damage} for c in self.hands[self.current_player]],
        }

    def get_valid_actions(self) -> List[Dict]:
        hand = self.hands[self.current_player]
        elixir = self.elixir[self.current_player]
        return [
            {"card": c, "cost": c.cost, "damage": c.damage}
            for c in hand
            if c.cost <= elixir
        ]

    def play_action(self, action: Optional[Dict]):
        if action:
            c: Card = action["card"]
            player = self.current_player
            opp = 1 - player
            if c.cost <= self.elixir[player]:
                self.elixir[player] -= c.cost
                # spawn card on arena
                self.arena_cards.append({
                    "card": c,
                    "player": player,
                    "position": 0 if player == 0 else 10  # start positions
                })
        self.next_turn()

    def next_turn(self):
        self.turn += 1
        # regen elixir
        self.elixir[0] = min(10, self.elixir[0] + 1)
        self.elixir[1] = min(10, self.elixir[1] + 1)
        # move cards
        for c in self.arena_cards:
            if c["player"] == 0:
                c["position"] += 1
            else:
                c["position"] -= 1
        # deal damage if reached tower
        new_arena = []
        for c in self.arena_cards:
            if c["player"] == 0 and c["position"] >= 10:
                self.hp[1] -= c["card"].damage
            elif c["player"] == 1 and c["position"] <= 0:
                self.hp[0] -= c["card"].damage
            else:
                new_arena.append(c)
        self.arena_cards = new_arena
        # swap player
        self.current_player = 1 - self.current_player
        # check game over
        if self.hp[0] <= 0:
            self.winner = 1
        elif self.hp[1] <= 0:
            self.winner = 0
        elif self.turn >= self.max_turns:
            if self.hp[0] > self.hp[1]:
                self.winner = 0
            elif self.hp[1] > self.hp[0]:
                self.winner = 1
            else:
                self.winner = None

    def is_over(self):
        return self.winner is not None

    def get_winner(self):
        return self.winner

    def get_reward(self):
        if self.winner is None:
            return 0
        return 1 if self.winner == 0 else -1


# -------------------------
# Adapter
# -------------------------

class GameAdapter:
    def __init__(self, game: SimpleClashGame):
        self.game = game

    def clone(self):
        return GameAdapter(self.game.clone())

    def get_state(self):
        return self.game.get_state()

    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def apply_action(self, action):
        return self.game.play_action(action)

    def is_over(self):
        return self.game.is_over()

    def get_reward(self):
        return self.game.get_reward()


# -------------------------
# Agents
# -------------------------

class HeuristicAgent:
    def __init__(self, adapter: GameAdapter):
        self.adapter = adapter

    def act(self):
        actions = self.adapter.get_valid_actions()
        if not actions:
            return None
        return max(actions, key=lambda a: a["damage"] / a["cost"])


class MCTSNode:
    def __init__(self, parent, action, adapter: GameAdapter):
        self.parent = parent
        self.action = action
        self.adapter = adapter
        self.children: List["MCTSNode"] = []
        self._untried: List[Dict] = adapter.get_valid_actions()[:]
        self.visits = 0
        self.value = 0.0

    def expand(self):
        if not self._untried:
            return None
        action = self._untried.pop(random.randrange(len(self._untried)))
        child_adapter = self.adapter.clone()
        child_adapter.apply_action(action)
        child = MCTSNode(self, action, child_adapter)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return not self._untried

    def best_child(self, c_param=1.25):
        if not self.children:
            return None
        choices_weights = [
            (c.value / (c.visits + 1e-9)) +
            c_param * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-9))
            for c in self.children
        ]
        return self.children[max(range(len(self.children)), key=lambda i: choices_weights[i])]

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


class MCTSAgent:
    def __init__(self, adapter: GameAdapter, iterations=100, time_budget=None):
        self.adapter = adapter
        self.iterations = iterations
        self.time_budget = time_budget

    def rollout(self, adapter: GameAdapter, depth=20):
        sim = adapter.clone()
        steps = 0
        while not sim.is_over() and steps < depth:
            actions = sim.get_valid_actions()
            if not actions:
                break
            a = random.choice(actions)
            sim.apply_action(a)
            steps += 1
        return sim.get_reward()

    def search(self):
        root = MCTSNode(None, None, self.adapter.clone())
        start = time.time()
        iters = 0
        while True:
            if self.time_budget and time.time() - start > self.time_budget:
                break
            if not self.time_budget and iters >= self.iterations:
                break

            node = root
            while node.children and node.is_fully_expanded():
                next_node = node.best_child()
                if next_node is None:
                    break
                node = next_node
            if node and node._untried:
                node = node.expand() or node
            reward = self.rollout(node.adapter)
            node.backpropagate(reward)
            iters += 1

        if not root.children:
            return None
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def act(self):
        return self.search()


# -------------------------
# Arena Visualization
# -------------------------

def draw_arena(game: SimpleClashGame):
    plt.clf()
    plt.xlim(-1, 11)
    plt.ylim(-1, 2)
    # Towers
    plt.plot(0, 0.5, 's', markersize=30, color='blue')
    plt.plot(10, 0.5, 's', markersize=30, color='red')
    # HP text
    plt.text(0, 1, f"HP: {game.hp[0]}", color='blue', fontsize=12, ha='center')
    plt.text(10, 1, f"HP: {game.hp[1]}", color='red', fontsize=12, ha='center')
    # Cards
    for c in game.arena_cards:
        color = 'green' if c["player"] == 0 else 'orange'
        plt.plot(c["position"], 0.5, 'o', markersize=15, color=color)
        plt.text(c["position"], 0.7, c["card"].name, fontsize=8, ha='center')
    plt.title(f"Turn {game.turn}")
    plt.pause(0.5)


# -------------------------
# Example: Bot Match in Arena
# -------------------------

def play_match():
    game = SimpleClashGame()
    adapter = GameAdapter(game)
    bot1 = MCTSAgent(adapter, iterations=100)
    bot2 = HeuristicAgent(adapter)

    plt.ion()
    plt.figure(figsize=(10,2))

    while not game.is_over():
        if game.current_player == 0:
            action = bot1.act()
        else:
            action = bot2.act()
        game.play_action(action)
        draw_arena(game)

    plt.ioff()
    draw_arena(game)
    print("Game Over. Winner:", game.get_winner(), "HP:", game.hp)
    plt.show()


if __name__ == "__main__":
    play_match()
