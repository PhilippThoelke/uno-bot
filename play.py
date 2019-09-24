import sys
import time
import pygame
import numpy as np
from keras.models import load_model
from environment import UnoEnvironment

MODEL_PATH = 'models/25-09-19_00-57-11/model-8000.h5'

CARD_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 150, 255), (255, 255, 0)]
CARD_WIDTH = 40
CARD_HEIGHT = 65
FONT = 'arial'
FONT_SIZE = 25

BACKGROUND = (180, 180, 180)
X_MARGIN = 5
Y_MARGIN = 10

POSSIBLE_PLAYER_TYPES = ['AI', 'Human', 'Safe']

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

# check command line arguments
if len(sys.argv) < 3:
    print(f'Not enough players specified ({len(sys.argv) - 1}).', end='')
    print(f'Select player types with command line attributes. (options: {POSSIBLE_PLAYER_TYPES})')
    exit()

# extract player types (0 for AI, 1 for human)
player_types = []
for player in sys.argv[1:]:
    found = False
    # find player type and add to list
    for i, possible in enumerate(POSSIBLE_PLAYER_TYPES):
        if player.lower() == possible.lower():
            player_types.append(i)
            found = True
            break

    if not found:
        # unknown player type found
        print(f'Unknown player type "{player}". Please select from {POSSIBLE_PLAYER_TYPES}.')
        exit()

# initialize pygame
print('Initializing pygame...')
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption(f'Uno game - {" vs. ".join([POSSIBLE_PLAYER_TYPES[i] for i in player_types])}')
clock = pygame.time.Clock()
font = pygame.font.SysFont(FONT, FONT_SIZE, bold=True)

if 0 in player_types:
    if MODEL_PATH is not None:
        print('Loading model...')
        model = load_model(MODEL_PATH)
    else:
        print('Please specify a model path.')
        exit()

print('Initializing game environment...')
env = UnoEnvironment(len(player_types))

if 1 in player_types:
    # TODO: remove value from player_types when player is eliminated
    # TODO: add "draw card" button
    print('WARNING! Human player not fully implemented yet.')

# init flags
mouse_down = False
clicked = False
done = False
game_finished = False
last_move = time.time()

print('Done! Running game loop...')
while not done:
    clicked = False
    # check events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                clicked = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = False


    if not game_finished:
        # reset screen
        screen.fill(BACKGROUND)
        card_rects = draw_env(env, screen, font)
        pygame.display.flip()

        if player_types[env.turn] == 0:
            state = env.get_state()
            action = np.argmax(model.predict(state.reshape((1, -1)))[0])
        elif player_types[env.turn] == 1:
            # human player's turn
            card_selected = False
            if clicked:
                mouse_pos = pygame.mouse.get_pos()
                index = np.argwhere([rect.contains(mouse_pos + (0, 0)) for rect in card_rects])
                if len(index) > 0:
                    action = np.argwhere(env.players[env.turn].cards > 0)[index[0,0],0]
                    if env.legal_move(action):
                        card_selected = True
            if not card_selected:
                # no card was selected
                continue
        elif player_types[env.turn] == 2:
            action = 0
            while not env.legal_move(action):
                action += 1

        # play the selected action
        game_finished = env.step(action)[-1]

        # update game screen once after game has finished
        if game_finished:
            screen.fill(BACKGROUND)
            draw_env(env, screen, font)
            pygame.display.flip()

    clock.tick(2)

pygame.quit()
