import numpy as np
import bpy

mb = bpy.data.texts["base.py"].as_module()
if __name__ == '__main__':
    cfg = {"name": "rectangle", "shape": {"Flange length": 1, "Web length": 3}}
    RC = mb.Rectangle(cfg, 2)
    RC.createBlenderObj("chenggong")
    cfg = {"name": "xxxx",
           "shape":
               {
               "length": 1,
                "width": 1
                }
           }