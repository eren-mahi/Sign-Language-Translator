import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Convolutional layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Convolutional layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data preparation
def prepare_data(train_dir, validation_dir, target_size, batch_size):
    # Image data generator for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Image data generator for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validat0ion data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def main():
    input_shape = (200, 200, 3)  # Image size and channels
    num_classes = 26  # Number of signs
    train_dir = 'D:/Sign Language Translator Using AI/Main/src/data/train'  # Path to training data
    validation_dir = 'D:/Sign Language Translator Using AI/Main/src/data/validation'  # Path to validation data
    target_size = (200, 200)  # Size of input images
    batch_size = 32  # Number of samples per gradient update

    model = create_model(input_shape, num_classes)

    train_generator, validation_generator = prepare_data(train_dir, validation_dir, target_size, batch_size)

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=20, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    test_dir = 'D:/Sign Language Translator Using AI/Main/src/data/test'  # Path to test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    model.save('models/gesture_model.h5')
    print("Model saved to 'models/gesture_model.h5'")

if __name__ == "__main__":
    main()
