
def deck_basic(w_deck, t_deck, h_deck):
    cfg = {
        "name": "deck_basic",
        "shape": {
            'deck width': w_deck,
            'deck thickness': t_deck,
            'deck top surface height': h_deck
        }
    }
    return cfg


def column_basic(w_column, h_column, t_column, h_deck, t_deck):
    cfg = {
        "name": "column_basic",
        "shape": {
            'column width': w_column,
            'column height': h_column,
            'column thickness': t_column,
            'deck top surface height': h_deck,
            'deck thickness': t_deck
        }
    }
    return cfg


def triangle(w, h, H):
    cfg = {
        "name": "triangle",
        "shape": {
            'bottom width': w,
            'height': h,
            'height of top': H
        }
    }
    return cfg


def triangle2(a1, b1, a2, b2, a3, b3, c):
    cfg = {
        "name": "triangle2",
        "shape": {
            'x1': a1,
            'x2': a2,
            'x3': a3,
            'z1': b1,
            'z2': b2,
            'z3': b3,
            'y': c
        }
    }
    return cfg


def rectangle(w, h, H):
    cfg = {
        "name": "rectangle",
        "shape": {
            'bottom width': w,
            'height': h,
            'height of top': H
        }
    }
    return cfg


def circle(coord, radius = 0.05, num = 5):
    cfg = {
        "name": "circle",
        "shape": {
            'coordinate of center': coord,
            'radius': radius,
            'element number': num
        }
    }
    return cfg