import sys
import pygame
import numpy as np
from environment import UnoEnvironment
from renderer import draw_card, draw_player, draw_env

MODEL_PATH = 'example_model.h5'

FONT = 'arial'
FONT_SIZE = 25

WINDOW_SIZE = (800, 600)
BACKGROUND = (180, 180, 180)
POSSIBLE_PLAYER_TYPES = ['AI', 'Human', 'Safe']

# check command line arguments
if len(sys.argv) < 3:
    print(f'Not enough players specified ({len(sys.argv) - 1}).', end='')
    player_options_str = ', '.join(POSSIBLE_PLAYER_TYPES)
    print(f'Select player types with command line attributes. (options: {player_options_str})')
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
        player_options_str = ', '.join(POSSIBLE_PLAYER_TYPES)
        print(f'Unknown player type "{player}". Please select from {player_options_str}.')
        exit()

# initialize pygame
print('Initializing pygame...')
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption(f'Uno game - {" vs. ".join([POSSIBLE_PLAYER_TYPES[i] for i in player_types])}')
clock = pygame.time.Clock()
font = pygame.font.SysFont(FONT, FONT_SIZE, bold=True)

if 0 in player_types:
    if MODEL_PATH is not None:
        print('Loading model...')
        from keras.models import load_model
        model = load_model(MODEL_PATH)
    else:
        print('Please specify a model path.')
        exit()

print('Initializing game environment...')
env = UnoEnvironment(len(player_types))

# init flags
mouse_down = False
clicked = False
done = False
game_finished = False

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
            # AI player
            state = env.get_state()
            action = np.argmax(model.predict(state.reshape((1, -1)))[0])
        elif player_types[env.turn] == 1:
            # human player
            card_selected = False
            if clicked:
                # check if one of the player's cards was selected
                mouse_pos = pygame.mouse.get_pos()
                index = np.argwhere([rect.contains(mouse_pos + (0, 0)) for rect in card_rects])
                if len(index) > 0:
                    if index[0,0] == np.sum(env.players[env.turn].cards):
                        # draw from stack selected
                        action = len(UnoEnvironment.CARD_TYPES)
                    else:
                        # one of the player's cards was clicked
                        cards = [[index] * int(count) for index, count in enumerate(env.players[env.turn].cards) if count > 0]
                        cards = np.concatenate(cards)
                        # get the selected card index
                        action = cards[index[0,0]]

                    if env.legal_move(action):
                        # set selected to true if the selected action is legal
                        card_selected = True
            if not card_selected:
                # no card was selected
                action = None
        elif player_types[env.turn] == 2:
            # safe player
            action = 0
            # search for first legal move
            while not env.legal_move(action):
                action += 1

        if action is not None:
            # play the selected action
            _, _, game_finished, step_info = env.step(action)
            last_move = time.time()

            turn = step_info['turn']
            player_status = step_info['player']

            # check if the current player is out of the game
            if player_status == -1 or player_status == 2:
                if player_status == -1:
                    game_messages.append((time.time(), f'{player_names[turn]} eliminated due to illegal move.'))
                elif player_status == 2:
                    game_messages.append((time.time(), f'{player_names[turn]} has finished!'))
                del player_types[turn]
                del player_names[turn]

            # update game screen once after game has finished
            if game_finished:
                screen.fill(BACKGROUND)
                draw_env(env, screen, font)
                pygame.display.flip()

    # limit the frame rate
    clock.tick(2)

pygame.quit()
