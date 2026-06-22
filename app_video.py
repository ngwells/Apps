import requests
from flask import Flask, Response, render_template_string
import time

app = Flask(__name__)

# XBotGo API endpoint that returns JSON with a fresh playUrl
PLAY_API_URL = "https://cloud.xbotgo.net/api/core/api/live/room/user/task/play/US/23459"

# Browser spoofing headers (critical for XBotGo)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Referer": "",
    "Origin": ""
}


def get_latest_play_url(retries=3, delay=1):
    """
    Calls the XBotGo API and extracts the latest HLS playUrl.
    Includes retry logic and logging.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Fetching playUrl (attempt {attempt})")
            r = requests.get(PLAY_API_URL, headers=BROWSER_HEADERS, timeout=5)
            r.raise_for_status()
            data = r.json()

            print("[DEBUG] API JSON:", data)

            # Adjust this if your JSON structure differs
            play_url = data["data"]["playUrl"]

            if play_url:
                print(f"[INFO] Got playUrl: {play_url}")
                return play_url

        except Exception as e:
            print(f"[ERROR] Failed to fetch playUrl: {e}")

        time.sleep(delay)

    raise RuntimeError("Unable to fetch playUrl after retries")


@app.route("/")
def index():
    """
    Serves a simple HTML page with an HLS.js video player.
    """
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>XBotGo Live Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            body {
                margin: 0;
                background: #0f0f0f;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            video {
                width: 90%;
                max-width: 900px;
                border-radius: 12px;
                background: #000;
            }
        </style>
    </head>
    <body>
        <video id="video" controls autoplay muted playsinline></video>

        <script>
            const video = document.getElementById("video");
            const streamURL = "/stream.m3u8";

            function loadStream() {
                if (Hls.isSupported()) {
                    const hls = new Hls({ maxBufferLength: 10 });
                    hls.loadSource(streamURL);
                    hls.attachMedia(video);
                } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
                    video.src = streamURL;
                }
            }

            loadStream();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/stream.m3u8")
def stream():
    """
    Proxies the XBotGo HLS playlist with:
    - Browser spoofing headers
    - Auto-refreshing playUrl
    - Retry logic
    - CORS headers
    """
    try:
        play_url = get_latest_play_url()
        print(f"[INFO] Proxying stream from: {play_url}")

        upstream = requests.get(
            play_url,
            headers=BROWSER_HEADERS,
            stream=True,
            timeout=10
        )

        if not upstream.ok:
            print(f"[ERROR] Upstream error: {upstream.status_code}")
            return Response("Upstream error", status=upstream.status_code)

    except Exception as e:
        print(f"[ERROR] Proxy error: {e}")
        return Response(f"Proxy error: {e}", status=500)

    def generate():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    resp = Response(generate(), content_type="application/vnd.apple.mpegurl")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
