[toc]
# BridgeGenerator

## Definition
In our definition, 
* [**Member**](BridgeGenerator/Member) is the basic structure part with some basic shapes, e.g. rectangle, circle, or I-shape. **(TBD)**
* **Component** can be simply divided into four major types -- superstructure, substructure, deck and surface feature. Each of them can be assembled by the basic members. **(TBD)**
* **Bridge** is the assemble of different components, which should be in the form of real bridge. **(TBD)**

## Code Structure
``` bash
├── BridgeGenerator
│   ├── Bridge
│   │   ├── __init__.py
│   │   └── bridge.py
│   ├── Components
│   │   ├── __init__.py
│   │   └── components.py
│   ├── Member
│   │   ├── __init__.py
│   │   ├── cfg.py
│   │   ├── member.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── main.py
│   ├── test_chj.py
│   └── test_rwh.py
```

## Package
### member
In [member.py](BridgeGenerator/Member/member.py), the superclass ```Member``` is defined as an abstract class that represents the common attributes and methods of each concrete "member".
|1|2|3|
|---|---|---|