import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. all_image_paths
all_image_paths = []
for category in ['class_0_non_agri', 'class_1_agri']:
    base = f'./images_dataSAT/{category}/'
    files = [os.path.join(base, f) for f in os.listdir(base)]
    all_image_paths.extend(files)

# 2. Bind paths/labels, print sample
all_labels = [0]*len(os.listdir('./images_dataSAT/class_0_non_agri')) + \
    [2]*len(os.listdir('./images_dataSAT/class_1_agri'))
binded = list(zip(all_image_paths, all_labels))
print(random.sample(binded, 5))

# 3. custom_data_generator
datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, zoom_range=0.2,
                             horizontal_flip=True, validation_split=0.2)
train_gen = datagen.flow_from_directory('./images_dataSAT',
                                        target_size=(64,64),
                                        batch_size=8,
                                        class_mode='binary',
                                        subset='training')

# 4. Validation data
val_gen = datagen.flow_from_directory('./images_dataSAT',
                                      target_size=(64,64),
                                      batch_size=8,
                                      class_mode='binary',
                                      subset='validation')
