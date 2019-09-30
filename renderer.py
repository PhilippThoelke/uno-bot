import pygame
import numpy as np
from environment import UnoEnvironment

CARD_COLOURS = [(255, 0, 0), # red
                (0, 255, 0), # green
                (0, 150, 255), # blue
                (255, 255, 0), # yellow
                (75, 75, 75)] # stack colour
CARD_WIDTH = 40
CARD_HEIGHT = 65
X_MARGIN = 5
Y_MARGIN = 10


def draw_card(pos, card, surface, font):
    # get card colour and type
    if type(card) == str:
        colour_index, card_type = 4, -1
    elif hasattr(card, '__iter__'):
        colour_index, card_type = card
    else:
        colour_index, card_type = UnoEnvironment.CARD_TYPES[card]

    # get card colour tuple
    colour = CARD_COLOURS[4]
    if colour_index is not None:
        colour = CARD_COLOURS[colour_index]

    # get string for the card text
    if card_type == -1:
        # stack
        card_text = '+'
    elif card_type < 10:
        # regular number
        card_text = str(card_type)
    elif card_type == 10:
        # invert direction
        card_text = '<-'
    elif card_type == 11:
        # 2+ card
        card_text = '2+'
    elif card_type == 12:
        # skip turn
        card_text = 'X'
    elif card_type == 13:
        # wishing card
        card_text = '?'
    elif card_type == 14:
        # 4+ card
        card_text = '4+'

    # draw card rectangle
    bounds = (pos[0] - CARD_WIDTH // 2, pos[1] - CARD_HEIGHT // 2, CARD_WIDTH, CARD_HEIGHT)
    surface.fill(colour, bounds)

    # draw card text
    text = font.render(card_text, True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = pos
    surface.blit(text, text_rect)

def draw_player(cards, has_turn, offset, surface, font):
    if has_turn:
        # current player has the turn
        x = offset[0] - CARD_WIDTH / 2 - X_MARGIN
        y = offset[1] - (CARD_HEIGHT + Y_MARGIN) / 2
        width = np.sum(cards) * (CARD_WIDTH + X_MARGIN) + X_MARGIN
        height = CARD_HEIGHT + Y_MARGIN
        # draw box behind the cards to indicate this player's turn
        surface.fill((255, 255, 255), (x, y, width, height))

    player_rects = []
    x, y = offset
    # go through all card types of which the player has more than zero
    for card_index in np.argwhere(cards > 0)[:,0]:
        # repeat drawing the card as many times as the player has the card
        for _ in range(cards[card_index].astype(int)):
            # draw the current card
            draw_card((x, y), card_index, surface, font)
            # add this card's bounds to the card rects list
            player_rects.append(pygame.rect.Rect((x - CARD_WIDTH / 2, y - CARD_HEIGHT / 2, CARD_WIDTH, CARD_HEIGHT)))
            # advance position
            x += CARD_WIDTH + X_MARGIN
    return player_rects

def draw_env(env, surface, font):
    x = CARD_WIDTH / 2 + X_MARGIN
    # draw top card
    y = surface.get_bounding_rect().center[1] - (CARD_HEIGHT + Y_MARGIN) / 2
    draw_card((x, y), env.top_card, surface, font)
    # draw card stack
    y = surface.get_bounding_rect().center[1] + (CARD_HEIGHT + Y_MARGIN) / 2
    draw_card((x, y), 'stack', surface, font)
    stack_rect = pygame.rect.Rect((x - CARD_WIDTH / 2, y - CARD_HEIGHT / 2, CARD_WIDTH, CARD_HEIGHT))

    card_rects = None
    # draw players' cards
    for i, player in enumerate(env.players):
        offset = (20 + CARD_WIDTH * 1.5, X_MARGIN + i * (CARD_HEIGHT + Y_MARGIN) + CARD_HEIGHT / 2)
        rects = draw_player(player.cards, i == env.turn, offset, surface, font)

        if i == env.turn:
            # current player has the turn
            card_rects = rects

    card_rects.append(stack_rect)
    return card_rects
