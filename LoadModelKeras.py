from keras.models import load_model
new_model = load_model('model_keras.h5')

new_model.summary()
new_model.get_weights()

