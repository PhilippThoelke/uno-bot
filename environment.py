import numpy as np


class UnoEnvironment:

    # generate all possible cards as tuples with the structure (colour:int, type:int)
    CARD_TYPES = np.array([[colour, type] for colour in range(4) for type in range(13)])

    def __init__(self, player_count):
        self.player_count = player_count
        self.reset()

    def reset(self):
        # initialize players
        self.players = [UnoPlayer(self) for _ in range(self.player_count)]

        # initialize card stack
        self.top_card = self.CARD_TYPES[np.random.randint(len(self.CARD_TYPES))]
        self.to_draw = 0

        # initialize turns
        self.turn = 0
        self.turn_direction = 1

    def step(self, action):
        reward = 0

        # retrieve current player instance
        player = self.players[self.turn]

        if self.top_card[1] == 11 and action < len(self.CARD_TYPES) and self.CARD_TYPES[action][1] != 11:
            # player has to draw cards due to 2+ card (player is not countering)
            player.draw_cards(self.to_draw)
            self.to_draw = 0

        if action == len(self.CARD_TYPES):
            # draw card action
            player.draw_cards(1)
            reward -= 1
        else:
            # try to play the selected card
            card = player.play_card(action)

            if card is not False:
                # legal move played
                reward += 1
                self.top_card = card

                if card[1] == 10:
                    # reverse direction card played
                    self.turn_direction *= -1
                elif card[1] == 11:
                    # 2+ card played
                    self.to_draw += 2
                elif card[1] == 12:
                    # skip turn card played
                    self.next_turn()

                if player.num_cards() == 0:
                    reward += 50
                    self.remove_player(player)
            else:
                # illegal move, eliminate player
                reward -= 10
                self.remove_player(player)

        # advance to the next turn
        self.next_turn()

        # generate state vector (current top card, own cards, amount to draw)
        states = [np.all(self.CARD_TYPES == self.top_card, axis=1).astype(np.float),
                  player.cards,
                  [self.to_draw]]
        state = np.concatenate(states)

        # check if the end of the episode was reached (only one player left)
        done = len(self.players) <= 1

        return state, reward, done

    def next_turn(self):
        # advance to the next turn
        self.turn += self.turn_direction

        # wrap turn index around when out of bounds
        if self.turn < 0:
            self.turn = len(self.players) - 1
        elif self.turn >= len(self.players):
            self.turn = 0

    def legal_move(self, card_index):
        # check if the last and current card's colour or type are the same
        card = self.CARD_TYPES[card_index]
        return self.top_card[0] == card[0] or self.top_card[1] == card[1]

    def remove_player(self, player):
        self.players.remove(player)
        if self.turn_direction == 1:
            self.turn -= 1

    def state_size(self):
        return len(self.CARD_TYPES) * 2 + 1

    def action_count(self):
        return len(self.CARD_TYPES) + 1


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

    def play_card(self, card_index):
        # check if the selected card exists in the player's hand and this move is legal
        if self.cards[card_index] == 0 or not self.game.legal_move(card_index):
            return False

        # play the selected card
        self.cards[card_index] -= 1
        return self.game.CARD_TYPES[card_index]

    def num_cards(self):
        return int(sum(self.cards))

    def __repr__(self):
        return f'UnoPlayer ({self.num_cards()})'
