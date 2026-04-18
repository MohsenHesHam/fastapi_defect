import numpy as np
import cv2
import tensorflow as tf
import base64

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

        # Draw annotations on a copy of the image
        annotated = (img_array * 255).astype(np.uint8).copy()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{pred_class_name} {confidence:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = bbox[0] if bbox else 10
        label_y = bbox[1] - 10 if bbox and bbox[1] > 20 else 20
        cv2.rectangle(annotated, (label_x, label_y - text_h - baseline),
                      (label_x + text_w, label_y + baseline), (0, 255, 0), -1)
        cv2.putText(annotated, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'class': pred_class_name,
            'confidence': round(float(confidence), 2),
            'defect_percentage': round(float(defect_pct), 2),
            'bbox': list(bbox) if bbox else None,
            'annotated_image': annotated_b64,
        }
    except Exception as e:
        raise ValueError(f"Error in defect detection: {str(e)}")