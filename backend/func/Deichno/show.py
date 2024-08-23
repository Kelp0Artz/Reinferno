import http.server
import socketserver
import webbrowser
import os

# Define the directory containing the HTML and CSS files
DIRECTORY = "backend/func/Deichno/pages/mainPage"
PORT = 8000
os.chdir(DIRECTORY)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    
    # Open the browser to the localhost at the specified port
    webbrowser.open(f"http://localhost:{PORT}/index.html")
    
    # Serve the content until the user interrupts
    httpd.serve_forever()
