import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

train_dir = r"C:\Users\siddh\Downloads\0"
test_dir = r"C:\Users\siddh\Downloads\0"


batch_size = 32
image_size = (224, 224)
num_classes = 10

def preprocess_image(image):

    image = tf.image.resize(image, image_size)

    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1./255)
val_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(image_size[0], image_size[1], 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=val_data)
