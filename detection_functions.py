import numpy as np
import cv2
import tensorflow as tf


def get_gradcam_heatmap(model, img_array, class_idx=None):
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer('out_relu')

    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)

        if class_idx is None:
            class_idx = tf.argmax(predictions[0])

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def heatmap_to_bbox(heatmap, original_img, threshold=0.4):
    img_h, img_w = original_img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
    binary_map = (heatmap_resized >= threshold).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    defect_area = cv2.contourArea(largest_contour)
    total_area = img_h * img_w
    defect_percentage = (defect_area / total_area) * 100

    return (x, y, w, h), defect_percentage


def detect_defect(model, img_array, class_names, threshold=0.4):
    try:
        pred = model.predict(img_array[np.newaxis, ...], verbose=0)
        pred_class_idx = np.argmax(pred[0])
        confidence = pred[0][pred_class_idx] * 100
        pred_class_name = class_names[pred_class_idx]
        heatmap = get_gradcam_heatmap(model, img_array[np.newaxis, ...], pred_class_idx)
        bbox, defect_pct = heatmap_to_bbox(heatmap, img_array, threshold)
        return {
            'class': pred_class_name,
            'confidence': round(float(confidence), 2),
            'defect_percentage': round(float(defect_pct), 2),
            'bbox': list(bbox) if bbox else None,
        }
    except Exception as e:
        raise ValueError(f"Error in defect detection: {str(e)}")