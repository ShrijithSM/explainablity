"""
Chronoscope Chat Launcher

Starts the backend server and opens the chat interface in the browser.
Run this file to begin the live chat + analysis session.

    python integration_hub/backend/launch_chat_system.py
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path


def main():
    print("=" * 70)
    print("CHRONOSCOPE CHAT SYSTEM - LAUNCHER")
    print("Standalone model interaction with live analysis streaming")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    backend_server = repo_root / "integration_hub" / "backend" / "chat_server.py"
    frontend_html = repo_root / "integration_hub" / "frontend" / "chronoscope_chat.html"
    
    print(f"[•] Repo root: {repo_root}")
    print(f"[•] Backend: {backend_server}")
    print(f"[•] Frontend: {frontend_html}")
    print()
    
    if not backend_server.exists():
        print(f"[!] Backend server not found at {backend_server}")
        sys.exit(1)
    
    if not frontend_html.exists():
        print(f"[!] Frontend HTML not found at {frontend_html}")
        sys.exit(1)
    
    # Start backend server
    print("[*] Starting backend server...")
    try:
        backend_process = subprocess.Popen(
            [sys.executable, str(backend_server)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print(f"[✓] Backend server started (PID: {backend_process.pid})")
    except Exception as e:
        print(f"[!] Failed to start backend: {e}")
        sys.exit(1)
    
    # Wait for server to be ready
    print("[•] Waiting for server to initialize (10s)...")
    time.sleep(10)
    
    # Check if server is running
    if backend_process.poll() is not None:
        stdout, stderr = backend_process.communicate()
        print(f"[!] Backend crashed immediately!")
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
        sys.exit(1)
    
    # Open frontend in browser
    frontend_url = f"file:///{frontend_html}".replace("\\", "/")
    print(f"[*] Opening frontend: {frontend_url}")
    
    try:
        webbrowser.open(frontend_url)
        print("[OK] Frontend opened in browser")
    except Exception as e:
        print(f"[!] Could not open browser: {e}")
        print(f"[•] Manually visit: {frontend_url}")
    
    print("\n" + "=" * 70)
    print("SYSTEM RUNNING")
    print("=" * 70)
    print(f"Backend:    http://127.0.0.1:8000")
    print(f"WebSocket:  ws://127.0.0.1:8000/ws/dashboard")
    print(f"Frontend:   {frontend_url}")
    print()
    print("Commands:")
    print("  * Type a prompt and press Enter to chat")
    print("  * Signals stream live to the dashboard on the right")
    print("  * Entropy & Velocity panels show real-time analysis")
    print("  * System log shows token generation & events")
    print()
    print("To stop: Close this terminal or press Ctrl+C")
    print("=" * 70 + "\n")
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print(f"\n[!] Backend process exited unexpectedly")
                stdout, stderr = backend_process.communicate()
                print(f"STDERR:\n{stderr}")
                break
    except KeyboardInterrupt:
        print(f"[*] Shutting down...")
        backend_process.terminate()
        time.sleep(2)
        if backend_process.poll() is None:
            backend_process.kill()
        print(f"[✓] System stopped")


if __name__ == "__main__":
    main()
