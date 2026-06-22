import requests
from flask import Flask, Response, render_template_string
from urllib.parse import urlparse, urljoin

app = Flask(__name__)

# === CONFIG: put your working .m3u8 URL here ===
UPSTREAM_PLAYLIST_URL = "https://prod-global-live-hls.xbotgo.net/prod-global-xbotgo-net/47AD567E8723410FAFE09F1283E69EB3.m3u8?sign=148b6f147bcc4b6f097299e1f7459532&t=1782257661&region=1"

# Optional: if XBotGo requires cookies, paste them here from DevTools
COOKIE_HEADER = ""  # e.g. "_ga=...; _fbp=...; _ga_00JWMJGQFQ=..."

BROWSER_HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Mobile Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "identity",
    "Connection": "keep-alive",
}


def make_headers(referer: str | None = None, add_range: bool = False) -> dict:
    headers = dict(BROWSER_HEADERS_BASE)
    if referer:
        headers["Referer"] = referer
    if add_range:
        headers["Range"] = "bytes=0-"
    if COOKIE_HEADER:
        headers["Cookie"] = COOKIE_HEADER
    return headers


def get_upstream_base_dir(playlist_url: str) -> str:
    parsed = urlparse(playlist_url)
    path = parsed.path
    base_dir = path.rsplit("/", 1)[0] + "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_dir}"


UPSTREAM_BASE_DIR = get_upstream_base_dir(UPSTREAM_PLAYLIST_URL)


@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>XBotGo Live Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            body { margin: 0; background: #0f0f0f; display: flex; justify-content: center; align-items: center; height: 100vh; }
            video { width: 90%; max-width: 900px; border-radius: 12px; background: #000; }
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
def stream_playlist():
    print(f"[INFO] Fetching playlist from: {UPSTREAM_PLAYLIST_URL}", flush=True)
    try:
        headers = make_headers(referer=UPSTREAM_PLAYLIST_URL)
        upstream = requests.get(UPSTREAM_PLAYLIST_URL, headers=headers, timeout=10)
        
        if upstream.status_code != 200:
            print(f"[ERROR] Upstream returned status code {upstream.status_code}", flush=True)
            print(f"[DEBUG] Response body: {upstream.text[:500]}", flush=True)
            return Response(f"Upstream server error: {upstream.status_code}", status=upstream.status_code)
            
        playlist_text = upstream.text
    except Exception as e:
        print(f"[ERROR] Connection failed to upstream playlist: {e}", flush=True)
        return Response(f"Playlist connection error: {e}", status=500)

    rewritten_lines = []
    for line in playlist_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and (".ts" in stripped or ".mp4" in stripped):
            if stripped.startswith("http://") or stripped.startswith("https://"):
                seg_name = stripped.rsplit("/", 1)[-1]
            else:
                seg_name = stripped
            rewritten_lines.append(f"/segment/{seg_name}")
        else:
            rewritten_lines.append(line)

    rewritten_playlist = "\n".join(rewritten_lines)
    resp = Response(rewritten_playlist, content_type="application/vnd.apple.mpegurl")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/segment/<path:filename>")
def stream_segment(filename):
    segment_url = urljoin(UPSTREAM_BASE_DIR, filename)
    try:
        headers = make_headers(referer=UPSTREAM_PLAYLIST_URL, add_range=True)
        upstream = requests.get(segment_url, headers=headers, stream=True, timeout=10)
    except Exception as e:
        print(f"[ERROR] Failed to fetch segment {filename}: {e}", flush=True)
        return Response(f"Segment error: {e}", status=500)

    if not upstream.ok:
        print(f"[ERROR] Upstream segment error {filename}: {upstream.status_code}", flush=True)
        return Response("Upstream segment error", status=upstream.status_code)

    def generate():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    resp = Response(generate(), content_type="video/mp2ts")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)