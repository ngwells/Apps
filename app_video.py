from flask import Flask, request, Response
import requests
from requests.exceptions import HTTPError

app = Flask(__name__)

@app.route('/stream.m3u8')
def proxy_stream():
    # 1. Dynamically grab the target XbotGo stream link from the query parameter
    target_url = request.args.get('url')
    
    if not target_url:
        return "Error: Missing required 'url' parameter. Usage: /stream.m3u8?url=YOUR_XBOTGO_URL", 400
        
    print(f"[INFO] Dynamically proxying stream request for: {target_url}")
    
    # 2. Safely attempt to fetch the raw playlist text from the remote media server
    try:
        # A 10-second timeout prevents Gunicorn workers from hanging infinitely
        response = requests.get(target_url, timeout=10)
        
        # This catches 500, 403, and 404 errors immediately instead of choking
        response.raise_for_status() 
        
        # 3. Return the raw .m3u8 text playlist with the proper streaming MIME type
        return Response(response.text, mimetype='application/x-mpegURL')
        
    except HTTPError as http_err:
        print(f"[ERROR] XbotGo remote media server returned an error: {http_err}")
        return "Streaming provider is currently unavailable (Expired link or stream ended)", 502
        
    except Exception as err:
        print(f"[ERROR] Unexpected error fetching playlist: {err}")
        return "Internal video proxy error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)