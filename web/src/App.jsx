import { useEffect, useRef, useState } from 'react';
import { Chessground } from 'chessground';
import { Chess, SQUARES } from 'chess.js';
import 'chessground/assets/chessground.base.css';
import 'chessground/assets/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';
import './App.css';

function App() {
  const containerRef = useRef(null);
  const cgRef = useRef(null);
  const gameRef = useRef(new Chess());
  
  // We use boardFen just to trigger React re-renders for the UI
  const [boardFen, setBoardFen] = useState(gameRef.current.fen());
  const [status, setStatus] = useState("Your turn (White)");
  const [isThinking, setIsThinking] = useState(false);
  const [logs, setLogs] = useState(["Board initializing..."]);

  function addLog(msg) {
    setLogs(prev => [new Date().toLocaleTimeString() + ": " + msg, ...prev].slice(0, 10));
  }

  // Global error handler
  useEffect(() => {
    const handleError = (e) => addLog("System Error: " + e.message);
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  // Initialize Chessground
  useEffect(() => {
    if (containerRef.current && !cgRef.current) {
      try {
        cgRef.current = Chessground(containerRef.current, {
          fen: gameRef.current.fen(),
          orientation: 'white',
          movable: {
            free: false,
            color: 'white',
            dests: getDests(gameRef.current)
          },
          events: {
            move: (orig, dest) => handlePlayerMove(orig, dest)
          },
          animation: { enabled: true, duration: 200 }
        });
        addLog("Board ready.");
      } catch (e) {
        addLog("Initialization Error: " + e.message);
      }
    }
  }, []);

  // Sync visual board when game state changes
  useEffect(() => {
    const game = gameRef.current;
    if (cgRef.current) {
      cgRef.current.set({
        fen: game.fen(),
        movable: {
          color: game.turn() === 'w' ? 'white' : 'black',
          dests: getDests(game)
        },
        turnColor: game.turn() === 'w' ? 'white' : 'black'
      });
    }

    if (game.turn() === 'b' && !game.isGameOver() && !isThinking) {
      const timer = setTimeout(fetchAIMove, 500);
      return () => clearTimeout(timer);
    }
    updateStatus();
  }, [boardFen]);

  function getDests(chess) {
    const dests = new Map();
    SQUARES.forEach(s => {
      const ms = chess.moves({ square: s, verbose: true });
      if (ms.length) dests.set(s, ms.map(m => m.to));
    });
    return dests;
  }

  function handlePlayerMove(orig, dest) {
    try {
      const move = gameRef.current.move({ from: orig, to: dest, promotion: 'q' });
      if (move) {
        addLog("You: " + move.san);
        setBoardFen(gameRef.current.fen());
      }
    } catch (e) {
      // Revert if somehow an illegal move got through CG
      cgRef.current.set({ fen: gameRef.current.fen() });
    }
  }

  async function fetchAIMove() {
    setIsThinking(true);
    setStatus("AI is thinking...");
    const currentFen = gameRef.current.fen();
    try {
      addLog("AI thinking...");
      const response = await fetch("http://127.0.0.1:5000/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: currentFen }),
      });
      const data = await response.json();
      if (data.move) {
        addLog("AI: " + (data.san || data.move));
        gameRef.current.move(data.move);
        setBoardFen(gameRef.current.fen());
      }
    } catch (error) {
      addLog("AI Error: " + error.message);
    } finally {
      setIsThinking(false);
    }
  }

  function updateStatus() {
    const game = gameRef.current;
    if (game.isCheckmate()) {
      setStatus("Checkmate! " + (game.turn() === 'w' ? "Black" : "White") + " wins.");
    } else if (game.isDraw()) {
      setStatus("Draw!");
    } else {
      setStatus(game.turn() === 'w' ? "Your turn (White)" : "AI's turn (Black)");
    }
  }

  function resetGame() {
    gameRef.current = new Chess();
    setBoardFen(gameRef.current.fen());
    setLogs(["Game reset."]);
    setStatus("Your turn (White)");
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Chess AI</h1>
        <p className={`status ${isThinking ? "thinking" : ""}`}>{status}</p>
        <div className="fen-debug">FEN: {boardFen}</div>
      </div>
      
      <div className="board-wrapper">
        <div ref={containerRef} className="cg-board" />
      </div>

      <div className="controls">
        <button onClick={resetGame}>New Game</button>
        <button onClick={async () => {
          addLog("Pinging backend...");
          try {
            const res = await fetch("http://127.0.0.1:5000/health");
            const d = await res.json();
            addLog("Backend OK: " + JSON.stringify(d));
          } catch (e) {
            addLog("Backend Error: " + e.message);
          }
        }} style={{ marginLeft: '10px', backgroundColor: '#3498db' }}>
          Test Connection
        </button>
      </div>

      <div className="debug-log">
        <h3>Debug Log</h3>
        <div className="log-list">
          {logs.map((log, i) => (
            <div key={i} className="log-item">{log}</div>
          ))}
        </div>
      </div>

      <div className="history">
        <h3>Move History</h3>
        <div className="history-list">
          {gameRef.current.history().map((move, i) => (
            <span key={i} className="history-item">
              {i % 2 === 0 ? `${Math.floor(i / 2) + 1}. ` : ""}
              {move} 
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
