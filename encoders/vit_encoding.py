from transformers import ViTImageProcessor, ViTModel


class VitEncoding(object):
    def __init__(self):
        # Build the model
        print("=> creating vit model")
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def embedding_extract(self, imgs):
        """
        :param: imgs: list, each is a numpy img with shape, h * w * 3

        returns:
            shape: len(imgs) * 768
        """
        inputs = self.vit_processor(images=imgs, return_tensors="pt")

        outputs = self.vit_model(**inputs)
        return outputs["pooler_output"]

        
if __name__ == "__main__":
    import numpy as np
    model = VitEncoding()
    imgs = [np.random.rand(224, 128, 3), np.random.rand(128, 234, 3), np.random.rand(111, 222, 3)]
    model.embedding_extract(imgs)
