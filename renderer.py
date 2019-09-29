import pygame
import numpy as np
from environment import UnoEnvironment

CARD_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 150, 255), (255, 255, 0)]
CARD_WIDTH = 40
CARD_HEIGHT = 65
X_MARGIN = 5
Y_MARGIN = 10


def draw_card(pos, card, surface, font):
    # get card colour and type
    if hasattr(card, '__iter__'):
        colour, card_type = card
    else:
        colour, card_type = UnoEnvironment.CARD_TYPES[card]

    # get string for the card text
    if card_type < 10:
        # regular number
        card_text = str(card_type)
    elif card_type == 10:
        # change direction
        card_text = '<-'
    elif card_type == 11:
        # draw two cards
        card_text = '2+'
    elif card_type == 12:
        # skip turn
        card_text = 'X'

    # draw card rectangle
    bounds = (pos[0] - CARD_WIDTH // 2, pos[1] - CARD_HEIGHT // 2, CARD_WIDTH, CARD_HEIGHT)
    surface.fill(CARD_COLOURS[colour], bounds)

    # draw card text
    text = font.render(card_text, True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = pos
    surface.blit(text, text_rect)

def draw_player(cards, has_turn, offset, surface, font):
    if has_turn:
        x = offset[0] - CARD_WIDTH / 2 - X_MARGIN
        y = offset[1] - (CARD_HEIGHT + Y_MARGIN) / 2
        width = np.sum(cards) * (CARD_WIDTH + X_MARGIN) + X_MARGIN
        height = CARD_HEIGHT + Y_MARGIN
        surface.fill((255, 255, 255), (x, y, width, height))

    player_rects = []
    x, y = offset
    for card_index in np.argwhere(cards > 0)[:,0]:
        draw_card((x, y), card_index, surface, font)
        player_rects.append(pygame.rect.Rect((x - CARD_WIDTH / 2, y - CARD_HEIGHT / 2, CARD_WIDTH, CARD_HEIGHT)))
        x += CARD_WIDTH + X_MARGIN
    return player_rects

def draw_env(env, surface, font):
    # draw top card
    pos = (CARD_WIDTH / 2 + X_MARGIN, surface.get_bounding_rect().center[1])
    draw_card(pos, env.top_card, surface, font)

    card_rects = None
    # draw players' cards
    for i, player in enumerate(env.players):
        offset = (20 + CARD_WIDTH * 1.5, X_MARGIN + i * (CARD_HEIGHT + Y_MARGIN) + CARD_HEIGHT / 2)
        rects = draw_player(player.cards, i == env.turn, offset, surface, font)

        if i == env.turn:
            # current player has the turn
            card_rects = rects
    return card_rects
