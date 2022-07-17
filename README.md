- [BridgeGenerator](#bridgegenerator)
  - [Definition](#definition)
  - [Code Structure](#code-structure)
  - [Package](#package)
    - [Member](#member)
      - [member's attributes](#members-attributes)
      - [member's methods](#members-methods)
    - [Component](#component)
    - [Bridge](#bridge)
  - [Tutorial](#tutorial)

# BridgeGenerator

## Definition
In our definition, 
* [**Member**](BridgeGenerator/Member) is the basic structure part with some basic shapes, e.g. rectangle, circle, or I-shape. **(TBD)**
* [**Component**](BridgeGenerator/Component) can be simply divided into four major types -- superstructure, substructure, deck and surface feature. Each of them can be assembled by the basic members. **(TBD)**
* [**Bridge**](BridgeGenerator/Bridge) is the assemble of different components, which should be in the form of real bridge. **(TBD)**

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
### Member
In [member.py](BridgeGenerator/Member/member.py), the superclass ```Member``` is defined as an abstract class that represents the common attributes and methods of each concrete "member".
#### member's attributes

<style>
table th:first-of-type {
    width: 30%;
}
table th:nth-of-type(2) {
    width: 20%;
}
table th:nth-of-type(3) {
    width: 50%;
}
</style>
We use cfg to define the name and shape of each different shape. The cfg file should be a dictionary with the following format.
```python
cfg = {
    'name': xxx,
    'shape': {
        'detail1': ,
        'detail2': ,
        ...
    }
}
```
As an example, the configuration for rectangle is
```python
rectangle_cfg = {
    'name': rectangle,
    'shape': {
        'Flange length': 2,
        'Web length': 1,
    }
}
```
The other attributes are given in the table.

Attribute|Data Type|Meaning
:-:|:-:|:-:
`name` | `str` |The name of the shape.
`shape`| `dict()` |A dictionary that includes all the parameters of the shape.
`yz`|2d `numpy.ndarray`|The coordination of one cross-section for the shape in the yz-plane.
`f`||The collection of the faces of the object.
`v`||The collection of the vertices of the object.
`n`||The number of cross-sections.
`t`||The translation for the cross-sections.
`r`||The rotation for the cross-sections.
`npts`|`int`|The number of points in one cross-section.
`obj`|`bpy.types.Object`|The corresponding blender object with the specific shape.

#### member's methods
<style>
table th:first-of-type {
    width: 20%;
}
table th:nth-of-type(2) {
    width: 20%;
}
table th:nth-of-type(3) {
    width: 60%;
}
</style>
Methods|Usage
:-:|:-:
`showCrossSection()`|An helper method returns the cross-sectional view of the object
`createObj(name)`|Create a blender object with the input `name`.
**should be some get and set methods here ...**|
### Component

### Bridge

## [Tutorial](BridgeGenerator/tutorial.md)