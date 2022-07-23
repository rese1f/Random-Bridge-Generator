# Tutorial
## 1. Blender Import Tutorial
### Preparation
Firstly, link all the python **files** you need in the blender text editor. 
### Implementation
For example, if you want to use the codes inside of the member.py in component.py, you need to add these codes at the beginning of the component.py.
```python
import bpy

# import the member.py as a module
mb = bpy.data.texts["member.py"].as_module()

# Then you can use the classes or functions inside of the member.py in the target python file
rectangle_cfg = {
    'name': rectangle,
    'shape': {
        'Flange length': 2,
        'Web length': 1,
    }
}
rectangle = mb.Rectangle(rectangle_cfg)
```