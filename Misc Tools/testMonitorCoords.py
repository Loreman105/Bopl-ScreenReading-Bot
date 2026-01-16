from screeninfo import get_monitors

for i, m in enumerate(get_monitors()):
    print(i, m.x, m.y, m.width, m.height)