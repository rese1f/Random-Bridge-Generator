# BridgeGenerator

## Cross Section

### Get start

```python
    from BridgeGenerator import Member

    if __name__ == '__main__':
        cfg = './BridgeGenerator/Member/configs/w_beam.yaml'
        w_beam = Member.wBeam(cfg)
```

### Add your own cross-section

1. Create a .py file and define a class in `./BridgeGenerator/CrossSection/`

```python
    from .base import Member

    class ClassName(Member):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.shape_parameter = ...

        def __call__(self):
            yz = ...
            return yz
```

2. Add your class in `./BridgeGenerator/CrossSection/__init__.py` to make sure it can be imported

```python
from .FileName import ClassName
```

3. Create a .yaml file and define the shape parameter in `./BridgeGenerator/CrossSection/configs/` as an example

```yaml
    name: wBeam
    shape: 
        Flange length: 1 #b
        Web length: 1 # h
        Flange thickness: 1 # tf
        Web thickness: 1 # tw
```