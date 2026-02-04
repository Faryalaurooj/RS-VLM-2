from utils.prompt_utils import build_prompts

def zero_shot_detect(model, image, class_name):
    prompts = build_prompts(class_name)
    text_embeds = encode_text(prompts)
    return model(image, text_embeds)

