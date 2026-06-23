from flask import Flask, Response
import requests

app = Flask(__name__)

# This is the permanent API link that provides the fresh .m3u8 streams
XBOTGO_API_URL = "https://cloud.xbotgo.net/api/core/api/live/room/user/task/play/US/23459"

@app.route('/stream.m3u8')
def proxy_stream():
    print("[INFO] Step 1: Querying XbotGo API for the live stream link...")
    
    try:
        # 1. Hit the API to get the dynamic stream data
        api_response = requests.get(XBOTGO_API_URL, timeout=10)
        api_response.raise_for_status()
        
        api_data = api_response.json()
        
        # 2. Extract the actual dynamic .m3u8 URL from the response payload
        real_stream_url = api_data.get('data', {}).get('playUrl')
        
        if not real_stream_url:
            print(f"[WARN] API worked, but no active stream found. Response: {api_data}")
            return "No active live stream found. Make sure the gimbal is broadcasting.", 503

        print(f"[INFO] Step 2: Extract successful! Fetching stream from: {real_stream_url}")

        # 3. Fetch the actual .m3u8 playlist file using the dynamic token
        stream_response = requests.get(real_stream_url, timeout=10)
        stream_response.raise_for_status()
        
        # 4. Pass the playlist text straight through to your platform video player
        return Response(stream_response.text, mimetype='application/x-mpegURL')
        
    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTP Error occurred: {http_err}")
        return "Streaming server error (Gimbal might be offline).", 502
    except ValueError:
        print("[ERROR] Failed to parse JSON from the XbotGo API.")
        return "Invalid response structure from backend.", 502
    except Exception as e:
        print(f"[ERROR] Unexpected proxy error: {str(e)}")
        return "Internal video proxy error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)