# Number of Input Channels
inChans = 4

# Input Image Shape
input_shape = (4, 160, 240, 240)

# Number of Output Classes
seg_outChans = 3

# The Activation Function to be used in the NVNet 
# options: [relu, elu]
activation = "relu"

# The Type of Normalization to be used in the NVNet 
# options: [group_normalization]
normalizaiton = "group_normalization"

# Whether to use VAE in the encoder-decoder network
VAE_enable = True

# The paths to the root folders of training images and training labels respectively
train_img_root = '/content/Task01_BrainTumour/imagesTr'
train_label_root = '/content/Task01_BrainTumour/labelsTr'

# The paths to the root folders of validation images and validation labels respectively
val_img_root = '/content/Task01_BrainTumour/imagesTr'
val_label_root = '/content/Task01_BrainTumour/labelsTr'

# Training and Validation Batch Size
train_batch_size = 1
val_batch_size = 1

# Path to Save the Training Weights
checkpoint_path = '/content'

# Number of Training Epochs
epochs = 100

# Initial Learning Rate
lr = 0.01