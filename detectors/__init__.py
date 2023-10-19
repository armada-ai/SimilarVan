from .mdef_detector import MdefDetector
from .yolov5_detector import Yolov5Detector
from .grid_detector import GridDetector


def choose_det_model(det_model_name, **kwargs):
    if det_model_name.lower() == "yolov5":
        model = Yolov5Detector(**kwargs)
    elif det_model_name.lower() == "mdef":
        model = MdefDetector(**kwargs)
    elif det_model_name.lower() == "grid":
        model = GridDetector()
    else:
        supported_models = ["yolov5", "mdef", "grid"]
        raise Error("only support %s" % supported_models)
    return model