import re

with open('dirty.txt', 'r') as f, open ('clean.txt', 'w') as f2:
    text = f.read()
    f.close()
    textreplace = re.sub('<.*?>', '', text)
    f2.write(textreplace)
    f2.close
