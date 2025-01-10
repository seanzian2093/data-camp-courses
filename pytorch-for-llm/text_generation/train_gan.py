"""Train a GAN model to generate text data."""

import torch
import torch.nn as nn
from build_gan import Generator, Discriminator

# Define the data
data = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)
# length of each synthetic data sequence
seq_length = 5

# total number of sequence generated
num_sequences = 100

# number of passes through the data
num_epochs = 50

# print loss every print_every epochs
print_every = 10

# Define the training data
generator = Generator(seq_length)
discriminator = Discriminator(seq_length)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for real_data in data:
        # Unsqueezing real_data and prevent gradient recalculations
        real_data = real_data.unsqueeze(0)
        noise = torch.rand((1, seq_length))
        fake_data = generator(noise)
        disc_real = discriminator(real_data)
        # `.detach()` ensures that the gradients calculated for the discriminator do not propagate back to the generator
        disc_fake = discriminator(fake_data.detach())
        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + criterion(
            disc_fake, torch.zeros_like(disc_fake)
        )
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # Train the generator
        disc_fake = discriminator(fake_data)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

    if (epoch + 1) % print_every == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs}:\t Generator loss: {loss_gen.item()}\t Discriminator loss: {loss_disc.item()}"
        )

print("\nReal data: ")
print(data[:5])

print("\nGenerated data: ")
for _ in range(5):
    noise = torch.rand((1, seq_length))
    generated_data = generator(noise)
    # Detach the tensor and print data
    print(torch.round(generated_data).detach())
