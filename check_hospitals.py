#utils.py:


from utils import get_unique_hospitals

h = get_unique_hospitals()

print("Found hospitals:", len(h))

for i, name in enumerate(h):
    print(i, name)