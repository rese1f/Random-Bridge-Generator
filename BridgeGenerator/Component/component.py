import bpy
mb = bpy.data.texts["member.py"].as_module()


class SubStructure:
    def __init__(self):
        self.structure = None

    def randomGenerate(self):
        structure_list = []


class SuperStructure:
    def __init__(self):
        self.structure = None


class Deck:
    def __init__(self):
        self.structure = None


class SurfaceFeature:
    def __init__(self):
        self.sturcture = None


class Cable(SuperStructure):
    def __init__(self):
        super.__init__()

