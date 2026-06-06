def simple(x):
    return x + 1


def tangled(a, b, c, d):
    total = 0
    if a > 0:
        for i in range(a):
            if b and i > 5:
                total += i
            elif c or d:
                total -= i
    elif a < 0:
        while b:
            b -= 1
            if c:
                total += 1
    else:
        total = 1 if d else 0
    return total
