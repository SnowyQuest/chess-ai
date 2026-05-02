# Chess AI Web Interface

This is a modern web interface for the PyTorch Chess AI.

## Prerequisites
- Python 3.x
- Node.js & npm
- A trained model checkpoint (e.g., `checkpoints/best_evolved.pt`)

## Setup & Running

### 1. Start the AI Backend
In the project root, run:
```bash
python server.py --model checkpoints/best.pt
```
The server will start on `http://localhost:5000`.

### 2. Start the React Frontend
In a new terminal, navigate to the `web` directory and run:
```bash
cd web
npm run dev
```
The interface will be available at the URL provided by Vite (usually `http://localhost:5173`).

## Features
- **Smooth Animations:** Powered by `react-chessboard`.
- **Legal Move Validation:** Handles all chess rules (castling, en passant, etc.) via `chess.js`.
- **Dark Mode:** A sleek, modern aesthetic.
- **AI Thinking Indicator:** Visual feedback when the AI is processing its move.
- **Move History:** Keeps track of all moves in the current game.
