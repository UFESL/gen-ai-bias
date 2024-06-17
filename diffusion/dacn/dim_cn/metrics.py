import torch
import torchmetrics.image
import torchmetrics.multimodal.clip_score

def get_eval_metrics(ground_truth, generated, prompt):
    inception = torchmetrics.image.inception.InceptionScore()
    fid = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)
    ground_truth = torch.stack(ground_truth, dim=0)
    generated = torch.stack(generated, dim=0)
    
    clip_score = torchmetrics.multimodal.clip_score.CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    prompt = [prompt] * len(generated)
    score = clip_score(generated, prompt)

    fid.update(ground_truth, real=True)
    fid.update(generated, real=False)
    return {"fid": fid.compute(), "clip": score.detach()}

