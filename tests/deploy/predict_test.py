import pytest

def test_predict(model, test_image_2d, test_image_3d):
    label = model.predict(test_image_2d)["pred_label"]

    assert label.shape == test_image_2d.shape

    assert label.max() == 22

    label = model.predict(test_image_2d, nms_iou=0.4)["pred_label"]

    assert label.max() == 22

    preds = model.predict(test_image_2d, output_type="contour")

    assert len(preds['pred_scores']) == 22
    assert len(preds['pred_contours']) == 22

    preds = model.predict(
        test_image_3d, 
        score_threshold=0.3,
        reshape_to=(60, 256, 256),
        nms_iou=0.4,
        output_type="contour",
    )

    assert len(preds['pred_scores']) > 0
    assert len(preds['pred_scores']) == len(preds['pred_contours'])


def test_f16_predict(model_f16, test_image_2d, test_image_3d):
    model = model_f16

    label = model.predict(test_image_2d)["pred_label"]

    assert label.shape == test_image_2d.shape

    assert label.max() == 22

    label = model.predict(test_image_2d, nms_iou=0.4)["pred_label"]

    assert label.max() == 22

    preds = model.predict(test_image_2d, output_type="contour")

    assert len(preds['pred_scores']) == 22
    assert len(preds['pred_contours']) == 22

    # preds = model.predict(
    #     test_image_3d, 
    #     score_threshold=0.3,
    #     reshape_to=(60, 256, 256),
    #     nms_iou=0.4,
    #     output_type="contour",
    # )

    # assert len(preds['pred_scores']) > 0
    # assert len(preds['pred_scores']) == len(preds['pred_contours'])
