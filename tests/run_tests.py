import sys
sys.path.insert(0, "src")
from test_chunking import test_chunk, test_brute, test_lsh, test_ivf

for fn in [test_chunk, test_brute, test_lsh, test_ivf]:
    try:
        fn()
        print(fn.__name__, "PASS")
    except Exception as e:
        print(fn.__name__, "FAIL:", e)
