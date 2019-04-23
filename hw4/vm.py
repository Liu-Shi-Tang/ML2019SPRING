from keras.models import Model, load_model
from keras.utils import plot_model

# read model
model_name = 'mcp-best-acc-0.68250.h5'
model = load_model(model_name)
model.summary()
# read data (label is not on-hot format)
# plot_model(model, to_file='model.png')





