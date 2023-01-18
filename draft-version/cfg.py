def setDeckBasic(w_deck, t_deck, h_deck):
    cfg = {
        "name": "deck_basic",
        "shape": {
            'deck width': w_deck,
            'deck thickness': t_deck,
            'deck top surface height': h_deck
        }
    }
    return cfg

def setBeamBasic(w_beam, h_beam, t_webplate, t_flange, h_deck, t_deck):
    cfg = {
        "name": "beam_basic",
        "shape": {
            'beam width': w_beam,
            'beam height': h_beam,
            'webplate thickness': t_webplate,
            'flange thickness': t_flange,
            'deck top surface height': h_deck,
            'deck thickness': t_deck       
        }
    }
    return cfg    

def setColumnBasic(w_column, h_column, t_column, h_deck, t_deck):
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

def setTriangle2(w, h, H):
    cfg = {
        "name": "triangle",
        "shape": {
            'bottom width': w,
            'height': h,
            'height of top': H
        }
    }
    return cfg

def setTriangle(a, b1, c1, b2, c2, b3, c3):
    cfg = {
        "name": "triangle2",
        "shape": {
            'y1': b1,
            'y2': b2,
            'y3': b3,
            'z1': c1,
            'z2': c2,
            'z3': c3,
            'x': a
        }
    }
    return cfg

def setRectangle(w, h, H):
    cfg = {
        "name": "rectangle",
        "shape": {
            'bottom width': w,
            'height': h, # (thickness in z-axis)
            'height of top': H
        }
    }
    return cfg

def setCircle(coord, radius = 0.05, num = 5):
    cfg = {
        "name": "circle",
        "shape": {
            'coordinate of center': coord,
            'radius': radius,
            'element number': num
        }
    }
    return cfg

def setPierCap(w,h,a,b,c):
    cfg = {
        "name": "PierCap",
        "shape": {
            'bottom width': w,
            'height': h, # height of deck - thickness of deck
            'a':a,
            'b':b,
            'c':c
        }
    }
    return cfg

def setBoxPierCap(w1,w2,w3,h,a,b):
    cfg = {
        "name": "PierCap",
        "shape": {
            'high width': w1,
            'mid width': w2,
            'low width': w3,
            'height': h, # height of deck - thickness of deck
            'a':a,
            'b':b
        }
    }
    return cfg

def setI_Beam(w,t,h,a,b):
    cfg = {
        "name": "PierCap",
        "shape": {
            'width': w,
            'thickness':t,
            'height': h, # height of deck - thickness of deck
            'a':a,
            'b':b
        }
    }
    return cfg

def setArchringBasic(w_archring, t_archring, h_archring):
    cfg = {
        "name": "archring_basic",
        "shape": {
            'archring width': w_archring,
            'archring thickness': t_archring,
            'archring top surface height': h_archring
        }
    }
    return cfg