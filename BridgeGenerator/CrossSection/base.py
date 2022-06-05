import yaml


class CrossSection():
    """
    define cross section in the local coordinate system
    x is the direction perpendicular to the cross-section
    cross-sections are defined in yz plane
    """

    def __init__(self, cfg):
        """load cfg

        Args:
            cfg (str): cfg file path
        """
        with open(cfg) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.name = self.cfg['name']
        self.shape =  self.cfg['shape']
        print("create {}\nshape {}".format(self.name, self.shape))