from .cvnet_encoding import CVNetEncoding
from .vit_encoding import VitEncoding


def choose_encoding_model(enconding_model_name, **kwargs):
    if enconding_model_name.lower() == "cvnet":
        model = CVNetEncoding(**kwargs)
    elif enconding_model_name.lower() == "vit":
        model = VitEncoding()
    else:
        supported_models = ["cvnet", "vit"]
        raise Error("only support %s" % supported_models)
    return model