from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
import requests

def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    if size > max_size:
        size = max_size

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  
    return image

def show_image(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()

def load_vgg19():
    vgg = vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False  
    return vgg

api_key = 'csk-69599dcnpnw3k63k9w6de4yve4p3t8yh5kpwx2pk36dt63p2'  
cerebras_url = 'https://api.cerebras.ai/inference'  

def run_inference_on_cerebras(content_image, style_image):
    data = {
        'content_image': content_image.squeeze(0).numpy().tolist(),
        'style_image': style_image.squeeze(0).numpy().tolist(),
        'model': 'vgg19'
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    response = requests.post(cerebras_url, json=data, headers=headers)

    if response.status_code == 200:
        result = torch.tensor(response.json()['stylized_image'])
        return result
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    content_image = load_image("/Users/sivansiri/Desktop/IMAGES/CITYSCAPE.jpeg")
    style_image = load_image("/Users/sivansiri/Desktop/IMAGES/MIRANDA.jpeg")
    style_image = transforms.Resize(content_image.shape[-2:])(style_image)

    print("Content Image:")
    show_image(content_image)

    print("Style Image:")
    show_image(style_image)
    stylized_image = run_inference_on_cerebras(content_image, style_image)

    if stylized_image is not None:
        print("Stylized Image:")
        show_image(stylized_image)
    else:
        print("Style transfer failed.")

