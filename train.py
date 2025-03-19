from gid_xpert import GID_Xpert
from data_preprocessing import load_dataset
import tensorflow as tf

train_gen, val_gen = load_dataset("dataset_path")

model = GID_Xpert((224, 224, 3), num_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=50)
model.save("GID_Xpert_model.h5")
