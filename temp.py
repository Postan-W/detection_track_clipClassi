from pathlib import Path
p = Path("./images/negative (1).jpg").resolve()
print(p.is_absolute())