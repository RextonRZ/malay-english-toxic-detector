
import torch
import torch.nn.functional as F

def predict_text(text, model, tokenizer, device, max_length=128):
    model.eval()

    encoding = tokenizer(
        str(text),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).squeeze()

    non_toxic_prob = probs[0].item()
    toxic_prob = probs[1].item()

    prediction_id = int(torch.argmax(probs).item())
    prediction_label = "Toxic" if prediction_id == 1 else "Non-toxic"
    confidence = max(non_toxic_prob, toxic_prob)

    return {
        "text": text,
        "prediction_id": prediction_id,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "non_toxic_probability": non_toxic_prob,
        "toxic_probability": toxic_prob
    }
