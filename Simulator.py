import random
from typing import Any, Dict, List, Optional, Tuple

SUITS = ["C", "D", "H", "S"]


# This is only used for determining for breaking Scuttle ties. Maybe move this down
def suit_rank(s: Optional[str]) -> int:
    if s is None:
        return 0
    return SUITS.index(s) + 1


class Card:
    def __init__(self, rank: int, suit: Optional[str]):
        self.rank = rank
        self.suit = suit

    def is_point(self) -> bool:
        return 1 <= self.rank <= 10

    def is_royal_or_glasses(self) -> bool:
        return self.rank == 8 or 11 <= self.rank <= 13

    def is_one_off_eligible(self) -> bool:
        return (1 <= self.rank <= 7) or self.rank == 9

    def display(self) -> str:
        rank_str = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"

    def copy(self):
        return Card(self.rank, self.suit)


class Deck:
    def __init__(self):
        self.cards: List[Card] = []
        for s in SUITS:
            for r in range(1, 14):
                self.cards.append(Card(r, s))

    def shuffle(self, rnd: random.Random):
        rnd.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        if not self.cards:
            return None
        return self.cards.pop(0)

    def size(self) -> int:
        return len(self.cards)

    def add_to_top(self, card: Card):
        self.cards.insert(0, card)


class PlayerState:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []
        self.point_field: List[Card] = []
        self.royals: List[Card] = []

    def copy(self):
        p = PlayerState(self.name)
        p.hand = [c.copy() for c in self.hand]
        p.point_field = [c.copy() for c in self.point_field]
        p.royals = [c.copy() for c in self.royals]
        return p

    # These are notable benefits of disjointing the field for players
    def points(self) -> int:
        return sum(c.rank for c in self.point_field)

    def count_kings(self) -> int:
        return sum(1 for c in self.royals if c.rank == 13)

    # Might need to change to zero for actually functionallity, cause 4 = win
    def effective_goal(self) -> int:
        k = self.count_kings()
        if k == 0:
            return 21
        if k == 1:
            return 14
        if k == 2:
            return 10
        if k == 3:
            return 5
        return 1

    def has_two(self) -> bool:
        return any(c.rank == 2 for c in self.hand)

    def remove_card_from_hand(self, idx: int) -> Card:
        return self.hand.pop(idx)


# Just does some comparisons to help check move legality
def can_scuttle(card: Card, target: Card) -> bool:
    if card is None or target is None:
        return False
    if not card.is_point() or not target.is_point():
        return False
    if card.rank > target.rank:
        return True
    if card.rank == target.rank:
        return suit_rank(card.suit) > suit_rank(target.suit)
    return False


# The game itself, we put the game loop back into the environment
# seed: seeds the randomization
# starting_hands: determines the size of the starting hands, makes things clearer inside the class
# agent0/agent1: actual players, instead of using a get_action method we just make agents callable
class Simulator:
    def __init__(
        self,
        seed: Optional[int] = None,
        starting_hands: Tuple[int, int] = (6, 5),
        agent0: Optional[callable] = None,
        agent1: Optional[callable] = None,
    ):
        # trunk-ignore(bandit/B311)
        self.rnd = random.Random(seed)
        self.starting_hands = starting_hands
        self.agents = [agent0, agent1]
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd.seed(seed)
        self.deck = Deck()
        self.deck.shuffle(self.rnd)
        self.scrap: List[Card] = []
        self.players = [PlayerState("Dealer"), PlayerState("Opponent")]
        self.current_player = 1  # player  (1) goes first per rules
        self.consecutive_passes = 0
        # deal
        for _ in range(self.starting_hands[0]):
            c = self.deck.draw()
            if c:
                self.players[0].hand.append(c)
        for _ in range(self.starting_hands[1]):
            c = self.deck.draw()
            if c:
                self.players[1].hand.append(c)
        return self.get_observation()

    # What is in an observation: TLDR a lot of lists, we leave it up to the agents to format it
    # Change needed, allow for certain indexes of opp.hand to be visible for opp_visible_hand
    def get_observation(self, for_player_index: Optional[int] = None) -> Dict[str, Any]:
        # Figure out which perspective we are "looking" from, defaults to current player.
        if for_player_index is None:
            for_player_index = self.current_player
        # Grab the actual player and their opponent
        p = self.players[for_player_index]
        opp = self.players[1 - for_player_index]
        # If there is an 8, hand is revealed (Need to add tracking for revealed cards)
        visible_opp_hand = None
        if any(c.rank == 8 for c in p.royals):
            visible_opp_hand = [c.copy() for c in opp.hand]
        # Create the observation using PlayerState classes
        obs = {
            "player_index": for_player_index,
            "hand": [c.copy() for c in p.hand],
            "own_point_field": [c.copy() for c in p.point_field],
            "own_royals": [c.copy() for c in p.royals],
            "opp_point_field": [c.copy() for c in opp.point_field],
            "opp_visible_hand": (
                [c.copy() for c in visible_opp_hand]
                if visible_opp_hand is not None
                else None
            ),
            "opp_point_count": len(opp.point_field),
            "opp_hand_size": len(opp.hand),
            "opp_royals_count": len(opp.royals),
            "visible_opp_hand": visible_opp_hand,
            "deck_count": self.deck.size(),
            "scrap_count": len(self.scrap),
            "whose_turn": self.current_player,
            "own_points": p.points(),
            "own_goal": p.effective_goal(),
            "opp_points": opp.points(),
            "opp_goal": opp.effective_goal(),
        }
        return obs
