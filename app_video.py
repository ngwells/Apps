from flask import Flask, Response, render_template_string
import requests

app = Flask(__name__)

# XBotGo API endpoint that returns JSON with a fresh playUrl
PLAY_API_URL = "https://cloud.xbotgo.net/api/core/api/live/room/user/task/play/US/23459"


def get_latest_play_url():
    """
    Calls the XBotGo API and extracts the latest HLS playUrl.
    """
    r = requests.get(PLAY_API_URL, timeout=5)
    r.raise_for_status()
    data = r.json()

    # Adjust this if your JSON structure differs
    return data["data"]["playUrl"]


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

            if (Hls.isSupported()) {
                const hls = new Hls();
                hls.loadSource(streamURL);
                hls.attachMedia(video);
            } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
                video.src = streamURL;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/stream.m3u8")
def stream():
    """
    Proxies the XBotGo HLS playlist with CORS headers.
    This keeps GitHub Pages and browsers happy.
    """
    try:
        play_url = get_latest_play_url()
        upstream = requests.get(play_url, stream=True, timeout=10)
    except Exception as e:
        return Response(f"Proxy error: {e}", status=500)

    if not upstream.ok:
        return Response("Upstream error", status=upstream.status_code)

    def generate():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    resp = Response(generate(), content_type="application/vnd.apple.mpegurl")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)