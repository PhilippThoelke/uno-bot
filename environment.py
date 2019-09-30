import numpy as np


class UnoEnvironment:

    ILLEGAL_MOVE_REWARD = -2
    DRAW_CARD_REWARD = -1
    CARD_PLAYED_REWARD = 2
    PLAYER_FINISHED_REWARD = 5

    # generate all possible cards as tuples with the structure (colour:int, type:int)
    CARD_TYPES = [[colour, type] for colour in range(4) for type in range(13)]
    CARD_TYPES += [[None, 13], [None, 14]]
    CARD_TYPES = np.array(CARD_TYPES)

    def __init__(self, player_count):
        self.player_count = player_count
        self.reset()

    def reset(self):
        # initialize players
        self.players = [UnoPlayer(self) for _ in range(self.player_count)]

        # initialize card stack
        self.top_card = self.CARD_TYPES[np.random.randint(len(self.CARD_TYPES))]
        if self.top_card[0] is None:
            self.top_card[0] = np.random.randint(0, 4)
        self.to_draw = 0

        # initialize turns
        self.turn = 0
        self.turn_direction = 1

    def step(self, action):
        reward = 0
        turn_index = self.turn

        # -1: illegal move, 0: draw card, 1: play card, 2: win
        player_status = None

        # retrieve current player instance
        player = self.players[self.turn]

        # get card selected by player (None => draw card)
        played_card = None
        if action < len(self.CARD_TYPES):
            played_card = self.CARD_TYPES[action].copy()

        if self.legal_move(action):
            if self.to_draw > 0:
                if played_card is None:
                    # player draw cards from previous 2+ or 4+ card(s)
                    player.draw_cards(self.to_draw)
                    self.to_draw = 0
                    player_status = 0
                elif self.top_card[1] == 11 and played_card[1] == 11:
                    # player adds 2+ card to existing
                    self.to_draw += 2
                    player_status = 1
                elif self.top_card[1] == 14 and played_card[1] == 14:
                    # player adds 4+ card to existing
                    self.to_draw += 4
                    player_status = 1
            elif played_card is None:
                # draw one card
                player.draw_cards(1)
                player_status = 0
            elif played_card[1] == 10:
                # reverse direction card
                self.turn_direction *= -1
                player_status = 1
            elif played_card[1] == 11:
                # 2+ card
                self.to_draw = 2
                player_status = 1
            elif played_card[1] == 12:
                # skip turn card
                self._next_turn()
                player_status = 1
            elif played_card[1] == 13:
                # wishing card
                played_card[0] = np.random.randint(0, 4)
                player_status = 1
            elif played_card[1] == 14:
                # 4+ card
                self.to_draw = 4
                played_card[0] = np.random.randint(0, 4)
                player_status = 1
            else:
                # play ordinary (0-9) card
                player_status = 1
        else:
            # illegal move, player eliminated
            player_status = -1

        if player_status == -1:
            reward += self.ILLEGAL_MOVE_REWARD
            self._remove_player(player)
        elif player_status == 0:
            reward += self.DRAW_CARD_REWARD
        elif player_status == 1:
            reward += self.CARD_PLAYED_REWARD
            player.play_card(action, played_card[0])

        if player.num_cards() == 0:
            # player has no cards left -> win
            player_status = 2
            reward += self.PLAYER_FINISHED_REWARD
            self._remove_player(player)

        if played_card is not None:
            # update top card with the card played by the player
            self.top_card = played_card

        # advance to the next turn
        self._next_turn()
        # check if the end of the episode was reached (only one player left)
        done = len(self.players) <= 1

        return self.get_state(), reward, done, {'turn': turn_index, 'player': player_status}

    def get_state(self):
        # generate state vector (current top card, own cards, amount to draw)
        states = [np.all(self.CARD_TYPES == self.top_card, axis=1).astype(np.float),
                  self.players[self.turn].cards,
                  [self.to_draw]]
        return np.concatenate(states)

    def _next_turn(self):
        # advance to the next turn
        self.turn += self.turn_direction

        # wrap turn index around when out of bounds
        if self.turn < 0:
            self.turn = len(self.players) - 1
        elif self.turn >= len(self.players):
            self.turn = 0

    def legal_move(self, action, player=None):
        if player is None:
            # get player that currently has the turn
            player = self.players[self.turn]

        # drawing a card is always legal
        if action == len(self.CARD_TYPES):
            return True

        # retrieve selected card information
        card = self.CARD_TYPES[action]

        # illegal move if the current player does not have the selected card
        if player.cards[action] == 0:
            return False
        # check if player has to draw cards after 2+ or also plays 2+
        if self.to_draw > 0 and self.top_card[1] == 11 and card[1] != 11:
            return False
        # check if player has to draw cards after 4+ or also plays 4+
        if self.to_draw > 0 and self.top_card[1] == 14 and card[1] != 14:
            return False
        # if conditions so far were fulfilled it is always legal to play wishing or 4+ cards
        if card[1] == 13 or card[1] == 14:
            return True

        # return true if the last and current card's colour or type are equal
        return self.top_card[0] == card[0] or self.top_card[1] == card[1]

    def _remove_player(self, player):
        self.players.remove(player)
        if self.turn_direction == 1:
            self.turn -= 1

    def state_size(self):
        return len(self.CARD_TYPES) * 2 + 1

    def action_count(self):
        return len(self.CARD_TYPES) + 1

    def players_left(self):
        return len(self.players)


class UnoPlayer:

    def __init__(self, game, num_cards=7):
        self.game = game

        # randomly initialize the player's hand
        self.cards = np.zeros(len(self.game.CARD_TYPES), dtype=np.float)
        self.draw_cards(num_cards)

    def draw_cards(self, count):
        # draw cards randomly
        for _ in range(count):
            self.cards[np.random.randint(len(self.game.CARD_TYPES))] += 1

    def play_card(self, card_index, colour=None):
        # check if this move is legal
        if not self.game.legal_move(card_index, player=self):
            return False

        # play the selected card
        self.cards[card_index] -= 1

        # set the colour of the played card if the colour was provided (for wishing and 4+ cards)
        card = self.game.CARD_TYPES[card_index].copy()
        if colour is not None:
            card[0] = colour
        return card

    def num_cards(self):
        return int(sum(self.cards))

    def __repr__(self):
        return f'UnoPlayer({self.num_cards()})'
