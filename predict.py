# from model.model import cnn_model
import cv2


test_data = cv2.imread("Sample/53.JPG")
test_data = cv2.cvtColor(test_data, cv2.COLOR_RGB2GRAY)

test_data = cv2.resize(test_data, (256, 256))
test_data = test_data.reshape(1, 256, 256, 1)
test_data = test_data.astype("float32")
test_data /= 255
print(test_data.shape)



# model = cnn_model()
# model.load_weights("papaya_model_2021052353.h5")
# ans = model.predict(test_data)
# print(ans)