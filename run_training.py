from utility.settings import init, init_training
init()
init_training() 
from utility.settings import *

from unet_model.training.train import epoch
import unet_model.unet as unet
from unet_model.training import training_datasets
from utility import user_interface
from utility.transformation import training_transforms
from utility.plotting import plot_loss_points
from unet_model.losses import dice_loss
train_losses = []
val_losses = []

def train_loop(model:unet.UNet=UNET_MODEL, loss_fn=LOSS_FN, 
               optimizer=torch.optim.Adam, device=DEVICE_COMPUTE_PLATFORM, num_epochs=NUM_EPOCHS, num_class = 1,
               val_dataloader=None, train_dataloader=None):
    
    loss_fn = nn.BCEWithLogitsLoss()
    model, optimizer = unet.initialize_model_training(pretrained_weights=None, num_class = num_class)


    for epoch_ii in range(num_epochs):
        train_loss, val_loss = epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epoch_ii)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    

def train(model:unet.UNet=MODEL_PARAMS, image_path=IMAGE_PATH, label_path=LABEL_PATH, 
          loss_fn:callable=LOSS_FN, optimizer=torch.optim.Adam, 
          device=DEVICE_COMPUTE_PLATFORM, num_epochs = NUM_EPOCHS, transforms=None):
    
    # Train the model 
    transforms = training_transforms
    print(f'image path: { image_path}')
    train_dataloader, val_dataloader = training_datasets.get_train_data_loaders(image_path=image_path, label_path=label_path, transform_generator=transforms)
    user_interface.startTrainingLogoPrint()
    print(f"Training the model using {loss_fn}")
    train_loop(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, num_epochs=num_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    
    plot_loss_points(train_losses, val_losses)
    print("Training complete.")


train()



