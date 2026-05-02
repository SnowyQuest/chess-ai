import torch
import chess
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.model import ChessNet
from src.engine import select_move

app = Flask(__name__)
CORS(app)

# Global variables for model, device, and args
model = None
device = None
args = None

def load_model(model_path, res_blocks, channels):
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_residual_blocks=res_blocks, channels=channels).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/move", methods=["POST"])
def get_move():
    print(f"Received request for FEN: {request.json.get('fen')}")
    data = request.json
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "No FEN provided"}), 400
    
    try:
        board = chess.Board(fen)
        # Use args.depth from the global scope
        move = select_move(model, board, device=device, depth=args.depth)
        return jsonify({
            "move": move.uci(),
            "san": board.san(move)
        })
    except Exception as e:
        print(f"Error processing move: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=2, help="Search depth (default: 2)")
    args = parser.parse_args()
    
    load_model(args.model, args.res_blocks, args.channels)
    app.run(host="0.0.0.0", port=args.port)
