# Generate music from facial image
import torch
from PIL import Image
from torchvision import transforms
from models.a_lightcnn import A_LightCNN
from models.ea_musicvae import EA_MusicVAE

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open("sample_face.jpg")
image_tensor = transform(image).unsqueeze(0)

cnn = A_LightCNN()
vae = EA_MusicVAE()

_, emotion_embedding = cnn(image_tensor)
midi_input = torch.randn(1, 32, 256)
output = vae(midi_input, emotion_embedding)

print("Generated MIDI shape:", output.shape)