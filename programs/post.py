import requests

url = '192.168.5.5:8000'

genres = 'Electric Folk HipHop International Latin Metal Noise Pop Rock Punk'.split()
for g in genres:
	for filename in os.listdir(f'{g}')
	songname = f'{g}/{filename}'
	data = open(songname, 'rb')
	file = {'file' data}

res = requests.post(url, files=file)
print(res.json())