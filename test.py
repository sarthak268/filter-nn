import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from networks import SobelNet  
from dataset import get_gt_values


with open('config.json', 'r') as file:
    args = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_run_model(image_name):
    model = SobelNet(num_layers=args["num_layers"])
    model = model.to(device)
    weights_path = './saved_weights/model.pth'
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Load and preprocess the image
    image_path = f"./{image_name}"  
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args["img_size"], args["img_size"])),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)

    output_np = output.squeeze(0).squeeze(0).cpu().numpy()

    output_np = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
    output_image = Image.fromarray(output_np)
    output_image.save('./predicted_image.jpg')

    gt_output = get_gt_values(input_image.cpu().squeeze(0))
    gt_output_np = gt_output.squeeze(0).squeeze(0).cpu().numpy()

    gt_output_np = ((gt_output_np - gt_output_np.min()) / (gt_output_np.max() - gt_output_np.min()) * 255).astype(np.uint8)
    output_image_gt = Image.fromarray(gt_output_np)
    output_image_gt.save('./gt_image.jpg')


if (__name__ == '__main__'):
    image_name = "test_image.jpg"
    result = load_and_run_model(image_name)
    
