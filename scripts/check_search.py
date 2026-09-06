import urllib.request, json

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode('utf-8')
            print('URL:', url)
            print('STATUS', resp.status)
            try:
                print(json.dumps(json.loads(body), indent=2))
            except Exception:
                print(body[:1000])
    except Exception as e:
        print('URL:', url, 'ERROR', repr(e))

if __name__ == '__main__':
    base = 'http://127.0.0.1:8001'
    queries = [
        f'{base}/api/v1/search/documents?q=COR-251',
        f'{base}/api/v1/search/attributes?q=COR-251',
    ]
    for q in queries:
        fetch(q)
