import requests, re, json

q = "Cirez D Tigerstyle"
url = f"https://www.beatport.com/search?q={q}"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36", "Accept": "text/html"}
r = requests.get(url, headers=headers, timeout=15)
m = re.search(r'__NEXT_DATA__.*?>(.*?)</script>', r.text, re.DOTALL)
data = json.loads(m.group(1))

def find_tracks(obj, results):
    if isinstance(obj, dict):
        if "bpm" in obj and ("name" in obj or "title" in obj):
            results.append(obj)
        for v in obj.values():
            find_tracks(v, results)
    elif isinstance(obj, list):
        for item in obj:
            find_tracks(item, results)
    return results

tracks = find_tracks(data, [])
print(f"Tracks encontrados: {len(tracks)}")
for t in tracks[:5]:
    artists = t.get("artists", "?")
    name = t.get("name", "?")
    bpm = t.get("bpm")
    key = t.get("key")
    genre = t.get("genre")
    print(f"  {artists} - {name}")
    print(f"    BPM: {bpm} | Key: {key} | Genre: {genre}")