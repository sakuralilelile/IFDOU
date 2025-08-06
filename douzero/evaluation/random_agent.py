import random

class RandomAgent():

    def __init__(self):
        self.name = 'Random'
        self.flag = True

    def act(self, infoset):
        if self.flag:
            # self.player_position = player_position
            # # The hand cands of the current player. A list.
            # self.player_hand_cards = None
            # # The number of cards left for each player. It is a dict with str-->int
            # self.num_cards_left_dict = None
            # # The three landload cards. A list.
            # self.three_landlord_cards = None
            # # The historical moves. It is a list of list
            # self.card_play_action_seq = None
            # # The union of the hand cards of the other two players for the current player
            # self.other_hand_cards = None
            # # The legal actions for the current move. It is a list of list
            # self.legal_actions = None
            # # The most recent valid move
            # self.last_move = None
            # # The most recent two moves
            # self.last_two_moves = None
            # # The last moves for all the postions
            # self.last_move_dict = None
            # # The played cands so far. It is a list.
            # self.played_cards = None
            # # The hand cards of all the players. It is a dict.
            # self.all_handcards = None
            # # Last player position that plays a valid move, i.e., not `pass`
            # self.last_pid = None
            # # The number of bombs played so far
            # self.bomb_num = None
            print("player_position: ", infoset.player_position)
            print("player_hand_cards: ", infoset.player_hand_cards)
            print("num_cards_left_dict: ", infoset.num_cards_left_dict)
            print("three_landlord_cards: ", infoset.three_landlord_cards)
            print("card_play_action_seq: ", infoset.card_play_action_seq)
            print("other_hand_cards: ", infoset.other_hand_cards)
            print("legal_actions: ", infoset.legal_actions)
            print("last_move: ", infoset.last_move)
            print("last_two_moves: ", infoset.last_two_moves)
            print("last_move_dict: ", infoset.last_move_dict)
            print("played_cards: ", infoset.played_cards)
            print("all_handcards: ", infoset.all_handcards)
            print("last_pid: ", infoset.last_pid)
            print("bomb_num: ", infoset.bomb_num)
            print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        return random.choice(infoset.legal_actions)
