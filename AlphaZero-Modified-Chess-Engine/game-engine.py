import pygame as p
from chess_board import board
from alpha_net import ChessNet
import encoder_decoder as ed
from threading import Thread
import time
import numpy as np
import torch
from MCTS_chess import do_decode_n_move_pieces, UCT_search
import os

size, fps, sq_size, images = 512, 15, 64, {}

def main():
    p.init()
    screen = p.display.set_mode((size, size))
    clock = p.time.Clock()
    screen.fill(p.Color('white'))

    loadImages()
    running = True
    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif b.player == 1: # -> if black move it's the computer turn, wait for them to
                # Start the computation in a separate thread
                computation_thread = Thread(target=chess_engine_move)
                computation_thread.start()
            elif e.type == p.MOUSEBUTTONDOWN:
                x, y = p.mouse.get_pos()
                col = x // sq_size
                row = y // sq_size
                registerClick(row, col, b)     
   
        drawGameState(screen, b)
        clock.tick(fps)
        p.display.flip()

def loadImages():
    pieces = ["wP", "wR", "wN", "wB", "wK", "wQ", "bP", "bR", "bN", "bB", "bK", "bQ"]
    for piece in pieces:
        images[piece] = p.transform.scale(p.image.load(f"images/{piece}.png"), (sq_size, sq_size))

def drawGameState(screen, board):
    drawBoard(screen)

    if board.clicked_cell:
        highlightCell(board.clicked_cell[0], board.clicked_cell[1], screen)
    
    for pos in b.allowed_moves:
        dotCell(pos[0], pos[1], screen)

    drawPieces(screen, board)
    

def drawBoard(screen):
    colors = [p.Color(240,217,181, 2), p.Color(181,136,99, 25)]
    for r in range(8):
        for c in range(8):
            color = colors[((r+c)%2)]
            p.draw.rect(screen, color, p.Rect(c*sq_size, r*sq_size, sq_size, sq_size))

def drawPieces(screen, board):
    for r in range(8):
        for c in range(8):
            color = 'b' if board.current_board[r][c].islower() else 'w'
            piece_type = board.current_board[r][c]
            if piece_type != " ":
                screen.blit(images[f"{color}{piece_type.upper()}"], (c * sq_size, r * sq_size))

def getPieceByPosition(row, col, board):    
    return board.current_board[row][col]

def registerClick(row, col, board):
    piece = getPieceByPosition(row, col, board)
    # Here must check if clicked in an allowed move with a piece selected, if so, move
    if board.clicked_cell:
        # intention to move
        allowed_moves = allowedMoves(board.clicked_cell)
        if (row, col) in allowed_moves:
            b.move_piece(b.clicked_cell, (row, col)) # move piece
            b.allowed_moves = [] # remove indication of allowed moves
            b.clicked_cell = None # remove indication of highlighted cell
        else:
            board.clicked_cell = None
            board.allowed_moves = []
    else: 
        is_your_own_piece = True if (board.player == 0 and piece.isupper()) or (board.player == 1 and piece.islower()) else False    
        board.clicked_cell = (row,col) if piece and is_your_own_piece and (row, col) != board.clicked_cell else None
        allowedMoves((row, col)) if is_your_own_piece else None

def allowedMoves(cell):
    i, j = cell
    piece = b.current_board[i][j]
    match piece:
        case "p":
            allowed_moves = b.move_rules_p(cell)[0] # returns tuple (moves, threats)
        case "P":
            allowed_moves = b.move_rules_P(cell)[0] # returns tuple (moves, threats)
        case "n":
            allowed_moves = b.move_rules_n(cell)
        case "N":
            allowed_moves = b.move_rules_N(cell)
        case "k":
            allowed_moves = b.move_rules_k() # takes no position
        case "K": 
            allowed_moves = b.move_rules_K() # takes no position
        case "r":
            allowed_moves = b.move_rules_r(cell)
        case "R":
            allowed_moves = b.move_rules_R(cell)
        case "q":
            allowed_moves = b.move_rules_q(cell)
        case "Q":
            allowed_moves = b.move_rules_Q(cell)
        case "b":
            allowed_moves = b.move_rules_b(cell)
        case "B":
            allowed_moves = b.move_rules_B(cell)

    b.allowed_moves = allowed_moves
    return allowed_moves

def highlightCell(row, col, screen):
    p.draw.rect(screen, p.Color(135,152,106,2), p.Rect(col*sq_size, row*sq_size, sq_size, sq_size))

def dotCell(row, col, screen):
    p.draw.circle(screen, p.Color(135,152,106,2), (col*sq_size+1/2*sq_size, row*sq_size+1/2*sq_size), 10.0)

def chess_engine_move():
    global chess_ai_running
    if not chess_ai_running:
        chess_ai_running = True
        run_Alpha_Zero()
        chess_ai_running = False
        
def run_Alpha_Zero():
    best_move, root = UCT_search(b, 777, net)
    do_decode_n_move_pieces(b,best_move) # decode move and move piece(s)

    

if __name__ == "__main__":
    chess_ai_running = False
    b = board()

    # AlphaZero ChessNet initialization
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    net_to_play="current_net_trained8_iter1.pth.tar"
    current_net_filename = os.path.join("./model_data/", net_to_play)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    main()