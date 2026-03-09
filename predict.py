import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.models as models

classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy"
]

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load(
    "model/plant_disease_model.pth", map_location="cpu"))
model.eval()

__all__ = ["predict", "model"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image):

    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)


def predict(image):

    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]
    confidence_score = round(confidence.item() * 100, 2)

    result = {
        "prediction": prediction,
        "confidence": confidence_score
    }

    # agar confidence low ho
    if confidence.item() < 0.6:
        result["warning"] = "Model is not confident. Try clearer leaf image."

    return result
