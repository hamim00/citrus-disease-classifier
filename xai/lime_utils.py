import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

def lime_explain(predict_fn, pil_img, img_size=224, top_labels=1, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()
    img_np = np.array(pil_img.resize((img_size, img_size)))  # HxWx3 uint8
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )
    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=10, hide_rest=False
    )
    vis = mark_boundaries(temp/255.0, mask)
    vis = (vis * 255).astype(np.uint8)
    return Image.fromarray(vis), label
