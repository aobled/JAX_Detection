import chess
import chess.engine
import tkinter as tk
from tkinter import messagebox
from io import BytesIO
from PIL import Image, ImageTk
import cairosvg
import os
import shutil

from chess_model_inference import load_chess_model, select_model_move

# Checkpoint par defaut pour le mode Humain (Blancs) vs Modele (Noirs) - copie figee
# (voir Dev Notes de la session : ne jamais pointer directement vers best_model_chess.pkl
# a la racine du repo, qui reste ecrit en continu par un entrainement en cours).
DEFAULT_MODEL_CHECKPOINT = "model_snapshot_chess.pkl"


class JeuEchecsTkinter:
    def __init__(self, root, vs_model=False, model_checkpoint=DEFAULT_MODEL_CHECKPOINT):
        self.root = root
        self.root.title("Échecs - Tkinter (optimisé)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.board = chess.Board()
        self.selected_square = None

        # Mode Humain (Blancs) vs Modele (Noirs), exploration hors perimetre Epic 9
        # (PRD Non-Goals) - le mode Humain vs Humain d'origine reste inchange si False.
        self.vs_model = vs_model
        self.chess_model = None
        self.chess_model_variables = None
        if self.vs_model:
            checkpoint_path = model_checkpoint if os.path.isabs(model_checkpoint) else os.path.join(self.script_dir, model_checkpoint)
            print(f"🤖 Chargement du modele echecs ({checkpoint_path})...")
            self.chess_model, self.chess_model_variables = load_chess_model(checkpoint_path)
            print("🤖 Modele charge - il jouera les Noirs.")
        self.cell_size = 500 // 8
        self.canvas = tk.Canvas(root, width=500, height=530)  # Augmente la hauteur pour la barre d'avantage
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # Initialisation des références d'images
        self.image_cache = {}  # Cache global pour les PhotoImage
        self.image_refs = []   # Liste permanente pour conserver les références des images
        self.load_all_piece_images()  # Précharge TOUTES les images une fois pour toute

        # Repli à 3 niveaux, aucun binaire copié dans le repo (112 Mo, on ne pollue pas
        # git) : (1) ./chess/stockfish si présent, (2) emplacement d'origine sur cette
        # machine (avant que chess_game.py ne soit copié dans ce projet), (3) PATH.
        local_stockfish = os.path.join(self.script_dir, "stockfish")
        original_location_stockfish = "/home/aobled/Desktop/Development/Chess/stockfish"
        if os.path.exists(local_stockfish):
            self.stockfish_path = local_stockfish
        elif os.path.exists(original_location_stockfish):
            self.stockfish_path = original_location_stockfish
        else:
            self.stockfish_path = shutil.which("stockfish")
        self.engine = None
        if self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                print(f"Erreur Stockfish: {e}")
        else:
            print("Stockfish introuvable : évaluation matérielle de secours utilisée.")

        self.cached_score = None
        self.refresh_evaluation()
        self.draw_board()  # Premier dessin

    def load_all_piece_images(self):
        """Précharge toutes les images des pièces et les stocke dans self.image_cache et self.image_refs."""
        svg_dir = os.path.join(self.script_dir, "SVG")

        piece_mapping = {
            'K': 'K.svg', 'Q': 'Q.svg', 'R': 'R.svg', 'B': 'B.svg', 'N': 'N.svg', 'P': 'P.svg',
            'k': 'k.svg', 'q': 'q.svg', 'r': 'r.svg', 'b': 'b.svg', 'n': 'n.svg', 'p': 'p.svg'
        }

        for symbol, svg_file in piece_mapping.items():
            svg_path = os.path.join(svg_dir, svg_file)
            if os.path.exists(svg_path):
                try:
                    png_bytes = BytesIO()
                    cairosvg.svg2png(url=svg_path, write_to=png_bytes)
                    png_bytes.seek(0)
                    img = Image.open(png_bytes)
                    img = img.resize((self.cell_size, self.cell_size), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    self.image_cache[symbol] = photo_img
                    self.image_refs.append(photo_img)  # Conserve la référence
                except Exception as e:
                    print(f"Erreur lors du chargement de {svg_path}: {e}")
                    img = Image.new('RGB', (self.cell_size, self.cell_size), color="red")
                    photo_img = ImageTk.PhotoImage(img)
                    self.image_cache[symbol] = photo_img
                    self.image_refs.append(photo_img)
            else:
                print(f"Fichier manquant: {svg_path}")
                img = Image.new('RGB', (self.cell_size, self.cell_size), color="red")
                photo_img = ImageTk.PhotoImage(img)
                self.image_cache[symbol] = photo_img
                self.image_refs.append(photo_img)

    def draw_board(self):
        """Dessine le damier et les pièces avec gestion robuste des références."""
        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                x1, y1 = col * self.cell_size, row * self.cell_size
                color = "white" if (row + col) % 2 == 0 else "#8B4513"
                self.canvas.create_rectangle(x1, y1, x1 + self.cell_size, y1 + self.cell_size, fill=color)

                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    symbol = piece.symbol()
                    if symbol in self.image_cache:
                        img = self.image_cache[symbol]
                        self.canvas.create_image(
                            x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                            image=img
                        )

        # Ajoute la barre d'avantage (utilise le score déjà calculé, sans relancer le moteur)
        if self.cached_score is not None:
            self.draw_advantage_bar(self.cached_score)

    def refresh_evaluation(self):
        """Recalcule et met en cache le score de la position actuelle."""
        self.cached_score = self.evaluate_position()

    def evaluate_position(self):
        """Évalue la position actuelle avec Stockfish ou une évaluation matérielle."""
        if self.engine:
            result = self.engine.analyse(self.board, chess.engine.Limit(depth=15))
            score = result["score"].white().score(mate_score=10000)
            return score / 100
        else:
            # Évaluation matérielle de secours
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            score = 0
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    score += value if piece.color == chess.WHITE else -value
            return score / 1

    def draw_advantage_bar(self, score):
        """Dessine une barre d'avantage sous le damier."""
        x0, y0 = 10, 520
        width, height = 480, 20
        max_score = 10

        self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, fill="gray")

        advantage_width = min(abs(score) / max_score * width, width)
        x = x0 if score < 0 else x0 + width - advantage_width
        color = "black" if score > 0 else "red"
        self.canvas.create_rectangle(x, y0, x + advantage_width, y0 + height, fill=color)

        text = f"{abs(score):.1f} pions" if abs(score) >= 0.1 else "Égalité"
        self.canvas.create_text(x0 + width // 2, y0 + height // 2, text=text, fill="white")

    def __del__(self):
        """Ferme Stockfish proprement à la fin."""
        if self.engine:
            self.engine.quit()

    def on_close(self):
        """Ferme le moteur avant de quitter, quand l'utilisateur ferme la fenêtre."""
        if self.engine:
            self.engine.quit()
            self.engine = None
        self.root.destroy()

    def on_click(self, event):
        """Gère les clics pour sélectionner/déplacer les pièces."""
        col = event.x // self.cell_size
        row = 7 - (event.y // self.cell_size)
        square = chess.square(col, row)

        if self.selected_square is None:
            if self.board.piece_at(square) and self.board.piece_at(square).color == self.board.turn:
                self.selected_square = square
                self.highlight_legal_moves(square)
        else:
            piece = self.board.piece_at(self.selected_square)
            promotion = None
            if piece and piece.piece_type == chess.PAWN and (
                (piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                (piece.color == chess.BLACK and chess.square_rank(square) == 0)
            ):
                promotion = self.ask_promotion_choice()

            move = chess.Move(self.selected_square, square, promotion=promotion)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.refresh_evaluation()
                self.draw_board()
                if self.board.is_game_over():
                    messagebox.showinfo("Fin de partie", f"Résultat: {self.board.result()}")
                elif self.vs_model and self.board.turn == chess.BLACK:
                    self.play_model_move()
            else:
                self.selected_square = None
                self.draw_board()

    def play_model_move(self):
        """Coup des Noirs joue par le modele echecs (mode vs_model uniquement)."""
        print("🤖 Le modele reflechit...")
        try:
            move, value = select_model_move(self.chess_model, self.chess_model_variables, self.board)
        except ValueError as e:
            print(f"⚠️ Le modele n'a pas pu choisir de coup: {e}")
            return
        print(f"🤖 Coup joue: {move} (value={value:.4f})")
        self.board.push(move)
        self.refresh_evaluation()
        self.draw_board()
        if self.board.is_game_over():
            messagebox.showinfo("Fin de partie", f"Résultat: {self.board.result()}")

    def ask_promotion_choice(self):
        """Ouvre une fenêtre modale pour choisir la pièce de promotion."""
        choice = {"piece": chess.QUEEN}
        dialog = tk.Toplevel(self.root)
        dialog.title("Promotion")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(dialog, text="Choisissez la pièce de promotion :").pack(padx=10, pady=10)

        options = [
            ("Dame", chess.QUEEN),
            ("Tour", chess.ROOK),
            ("Fou", chess.BISHOP),
            ("Cavalier", chess.KNIGHT),
        ]

        def select(piece_type):
            choice["piece"] = piece_type
            dialog.destroy()

        frame = tk.Frame(dialog)
        frame.pack(padx=10, pady=10)
        for label, piece_type in options:
            tk.Button(frame, text=label, width=10, command=lambda pt=piece_type: select(pt)).pack(side=tk.LEFT, padx=5)

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)
        return choice["piece"]

    def highlight_legal_moves(self, square):
        """Surligne les coups légaux."""
        self.draw_board()
        for move in self.board.legal_moves:
            if move.from_square == square:
                col_to = chess.square_file(move.to_square)
                row_to = 7 - chess.square_rank(move.to_square)
                x1, y1 = col_to * self.cell_size, row_to * self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x1 + self.cell_size, y1 + self.cell_size,
                    fill="lightgreen", stipple="gray25", outline=""
                )

    def start(self):
        """Démarre la boucle principale Tkinter."""
        self.root.mainloop()

if __name__ == "__main__":
    import sys

    # Usage: python3 chess/chess_game.py [--vs-model]
    # --vs-model : Humain (Blancs, clics) vs modele echecs (Noirs, chess_model_inference.py)
    # Sans argument : Humain vs Humain (comportement d'origine, inchange).
    vs_model = "--vs-model" in sys.argv

    root = tk.Tk()
    jeu = JeuEchecsTkinter(root, vs_model=vs_model)
    jeu.start()
